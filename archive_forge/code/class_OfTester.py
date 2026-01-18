import binascii
import inspect
import json
import logging
import math
import netaddr
import os
import signal
import sys
import time
import traceback
from random import randint
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import hub
from os_ken.lib import stringify
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_5
class OfTester(app_manager.OSKenApp):
    """ OpenFlow Switch Tester. """
    tester_ver = None
    target_ver = None

    def __init__(self):
        super(OfTester, self).__init__()
        self._set_logger()
        self.interval = CONF['test-switch']['interval']
        self.target_dpid = self._convert_dpid(CONF['test-switch']['target'])
        self.target_send_port_1 = CONF['test-switch']['target_send_port_1']
        self.target_send_port_2 = CONF['test-switch']['target_send_port_2']
        self.target_recv_port = CONF['test-switch']['target_recv_port']
        self.tester_dpid = self._convert_dpid(CONF['test-switch']['tester'])
        self.tester_send_port = CONF['test-switch']['tester_send_port']
        self.tester_recv_port_1 = CONF['test-switch']['tester_recv_port_1']
        self.tester_recv_port_2 = CONF['test-switch']['tester_recv_port_2']
        self.logger.info('target_dpid=%s', dpid_lib.dpid_to_str(self.target_dpid))
        self.logger.info('tester_dpid=%s', dpid_lib.dpid_to_str(self.tester_dpid))

        def __get_version(opt):
            vers = {'openflow10': ofproto_v1_0.OFP_VERSION, 'openflow13': ofproto_v1_3.OFP_VERSION, 'openflow14': ofproto_v1_4.OFP_VERSION, 'openflow15': ofproto_v1_5.OFP_VERSION}
            ver = vers.get(opt.lower())
            if ver is None:
                self.logger.error('%s is not supported. Supported versions are %s.', opt, list(vers.keys()))
                self._test_end()
            return ver
        target_opt = CONF['test-switch']['target_version']
        self.logger.info('target ofp version=%s', target_opt)
        OfTester.target_ver = __get_version(target_opt)
        tester_opt = CONF['test-switch']['tester_version']
        self.logger.info('tester ofp version=%s', tester_opt)
        OfTester.tester_ver = __get_version(tester_opt)
        ofproto_protocol.set_app_supported_versions([OfTester.target_ver, OfTester.tester_ver])
        test_dir = CONF['test-switch']['dir']
        self.logger.info('Test files directory = %s', test_dir)
        self.target_sw = OpenFlowSw(DummyDatapath(), self.logger)
        self.tester_sw = OpenFlowSw(DummyDatapath(), self.logger)
        self.state = STATE_INIT_FLOW
        self.sw_waiter = None
        self.waiter = None
        self.send_msg_xids = []
        self.rcv_msgs = []
        self.ingress_event = None
        self.ingress_threads = []
        self.thread_msg = None
        self.test_thread = hub.spawn(self._test_sequential_execute, test_dir)

    def _set_logger(self):
        self.logger.propagate = False
        s_hdlr = logging.StreamHandler()
        self.logger.addHandler(s_hdlr)
        if CONF.log_file:
            f_hdlr = logging.handlers.WatchedFileHandler(CONF.log_file)
            self.logger.addHandler(f_hdlr)

    def _convert_dpid(self, dpid_str):
        try:
            return int(dpid_str, 16)
        except ValueError as err:
            self.logger.error('Invarid dpid parameter. %s', err)
            self._test_end()

    def close(self):
        if self.test_thread is not None:
            hub.kill(self.test_thread)
        if self.ingress_event:
            self.ingress_event.set()
        hub.joinall([self.test_thread])
        self._test_end('--- Test terminated ---')

    @set_ev_cls(ofp_event.EventOFPStateChange, [handler.MAIN_DISPATCHER, handler.DEAD_DISPATCHER])
    def dispatcher_change(self, ev):
        assert ev.datapath is not None
        if ev.state == handler.MAIN_DISPATCHER:
            self._register_sw(ev.datapath)
        elif ev.state == handler.DEAD_DISPATCHER:
            self._unregister_sw(ev.datapath)

    def _register_sw(self, dp):
        vers = {ofproto_v1_0.OFP_VERSION: 'openflow10', ofproto_v1_3.OFP_VERSION: 'openflow13', ofproto_v1_4.OFP_VERSION: 'openflow14', ofproto_v1_5.OFP_VERSION: 'openflow15'}
        if dp.id == self.target_dpid:
            if dp.ofproto.OFP_VERSION != OfTester.target_ver:
                msg = 'Join target SW, but ofp version is not %s.' % vers[OfTester.target_ver]
            else:
                self.target_sw.dp = dp
                msg = 'Join target SW.'
        elif dp.id == self.tester_dpid:
            if dp.ofproto.OFP_VERSION != OfTester.tester_ver:
                msg = 'Join tester SW, but ofp version is not %s.' % vers[OfTester.tester_ver]
            else:
                self.tester_sw.dp = dp
                msg = 'Join tester SW.'
        else:
            msg = 'Connect unknown SW.'
        if dp.id:
            self.logger.info('dpid=%s : %s', dpid_lib.dpid_to_str(dp.id), msg)
        if not (isinstance(self.target_sw.dp, DummyDatapath) or isinstance(self.tester_sw.dp, DummyDatapath)):
            if self.sw_waiter is not None:
                self.sw_waiter.set()

    def _unregister_sw(self, dp):
        if dp.id == self.target_dpid:
            self.target_sw.dp = DummyDatapath()
            msg = 'Leave target SW.'
        elif dp.id == self.tester_dpid:
            self.tester_sw.dp = DummyDatapath()
            msg = 'Leave tester SW.'
        else:
            msg = 'Disconnect unknown SW.'
        if dp.id:
            self.logger.info('dpid=%s : %s', dpid_lib.dpid_to_str(dp.id), msg)

    def _test_sequential_execute(self, test_dir):
        """ Execute OpenFlow Switch test. """
        tests = TestPatterns(test_dir, self.logger)
        if not tests:
            self.logger.warning(NO_TEST_FILE)
            self._test_end()
        test_report = {}
        self.logger.info('--- Test start ---')
        test_keys = list(tests.keys())
        test_keys.sort()
        for file_name in test_keys:
            report = self._test_file_execute(tests[file_name])
            for result, descriptions in report.items():
                test_report.setdefault(result, [])
                test_report[result].extend(descriptions)
        self._test_end(msg='---  Test end  ---', report=test_report)

    def _test_file_execute(self, testfile):
        report = {}
        for i, test in enumerate(testfile.tests):
            desc = testfile.description if i == 0 else None
            result = self._test_execute(test, desc)
            report.setdefault(result, [])
            report[result].append([testfile.description, test.description])
            hub.sleep(self.interval)
        return report

    def _test_execute(self, test, description):
        if isinstance(self.target_sw.dp, DummyDatapath) or isinstance(self.tester_sw.dp, DummyDatapath):
            self.logger.info('waiting for switches connection...')
            self.sw_waiter = hub.Event()
            self.sw_waiter.wait()
            self.sw_waiter = None
        if description:
            self.logger.info('%s', description)
        self.thread_msg = None
        try:
            self._test(STATE_INIT_METER)
            self._test(STATE_INIT_GROUP)
            self._test(STATE_INIT_FLOW, self.target_sw)
            self._test(STATE_INIT_THROUGHPUT_FLOW, self.tester_sw)
            for flow in test.prerequisite:
                if isinstance(flow, self.target_sw.dp.ofproto_parser.OFPFlowMod):
                    self._test(STATE_FLOW_INSTALL, self.target_sw, flow)
                    self._test(STATE_FLOW_EXIST_CHK, self.target_sw.send_flow_stats, flow)
                elif isinstance(flow, self.target_sw.dp.ofproto_parser.OFPMeterMod):
                    self._test(STATE_METER_INSTALL, self.target_sw, flow)
                    self._test(STATE_METER_EXIST_CHK, self.target_sw.send_meter_config_stats, flow)
                elif isinstance(flow, self.target_sw.dp.ofproto_parser.OFPGroupMod):
                    self._test(STATE_GROUP_INSTALL, self.target_sw, flow)
                    self._test(STATE_GROUP_EXIST_CHK, self.target_sw.send_group_desc_stats, flow)
            for pkt in test.tests:
                if KEY_EGRESS in pkt or KEY_PKT_IN in pkt:
                    target_pkt_count = [self._test(STATE_TARGET_PKT_COUNT, True)]
                    tester_pkt_count = [self._test(STATE_TESTER_PKT_COUNT, False)]
                elif KEY_THROUGHPUT in pkt:
                    for throughput in pkt[KEY_THROUGHPUT]:
                        flow = throughput[KEY_FLOW]
                        self._test(STATE_THROUGHPUT_FLOW_INSTALL, self.tester_sw, flow)
                        self._test(STATE_THROUGHPUT_FLOW_EXIST_CHK, self.tester_sw.send_flow_stats, flow)
                    start = self._test(STATE_GET_THROUGHPUT)
                elif KEY_TBL_MISS in pkt:
                    before_stats = self._test(STATE_GET_MATCH_COUNT)
                if KEY_INGRESS in pkt:
                    self._one_time_packet_send(pkt)
                elif KEY_PACKETS in pkt:
                    self._continuous_packet_send(pkt)
                if KEY_EGRESS in pkt or KEY_PKT_IN in pkt:
                    result = self._test(STATE_FLOW_MATCH_CHK, pkt)
                    if result == TIMEOUT:
                        target_pkt_count.append(self._test(STATE_TARGET_PKT_COUNT, True))
                        tester_pkt_count.append(self._test(STATE_TESTER_PKT_COUNT, False))
                        test_type = KEY_EGRESS if KEY_EGRESS in pkt else KEY_PKT_IN
                        self._test(STATE_NO_PKTIN_REASON, test_type, target_pkt_count, tester_pkt_count)
                elif KEY_THROUGHPUT in pkt:
                    end = self._test(STATE_GET_THROUGHPUT)
                    self._test(STATE_THROUGHPUT_CHK, pkt[KEY_THROUGHPUT], start, end)
                elif KEY_TBL_MISS in pkt:
                    self._test(STATE_SEND_BARRIER)
                    hub.sleep(INTERVAL)
                    self._test(STATE_FLOW_UNMATCH_CHK, before_stats, pkt)
            result = [TEST_OK]
            result_type = TEST_OK
        except (TestFailure, TestError, TestTimeout, TestReceiveError) as err:
            result = [TEST_ERROR, str(err)]
            result_type = str(err).split(':', 1)[0]
        finally:
            self.ingress_event = None
            for tid in self.ingress_threads:
                hub.kill(tid)
            self.ingress_threads = []
        self.logger.info('    %-100s %s', test.description, result[0])
        if 1 < len(result):
            self.logger.info('        %s', result[1])
            if result[1] == OSKEN_INTERNAL_ERROR or result == 'An unknown exception':
                self.logger.error(traceback.format_exc())
        hub.sleep(0)
        return result_type

    def _test_end(self, msg=None, report=None):
        self.test_thread = None
        if msg:
            self.logger.info(msg)
        if report:
            self._output_test_report(report)
        pid = os.getpid()
        os.kill(pid, signal.SIGTERM)

    def _output_test_report(self, report):
        self.logger.info('%s--- Test report ---', os.linesep)
        error_count = 0
        for result_type in sorted(list(report.keys())):
            test_descriptions = report[result_type]
            if result_type == TEST_OK:
                continue
            error_count += len(test_descriptions)
            self.logger.info('%s(%d)', result_type, len(test_descriptions))
            for file_desc, test_desc in test_descriptions:
                self.logger.info('    %-40s %s', file_desc, test_desc)
        self.logger.info('%s%s(%d) / %s(%d)', os.linesep, TEST_OK, len(report.get(TEST_OK, [])), TEST_ERROR, error_count)

    def _test(self, state, *args):
        test = {STATE_INIT_FLOW: self._test_initialize_flow, STATE_INIT_THROUGHPUT_FLOW: self._test_initialize_flow, STATE_INIT_METER: self.target_sw.del_meters, STATE_INIT_GROUP: self.target_sw.del_groups, STATE_FLOW_INSTALL: self._test_msg_install, STATE_THROUGHPUT_FLOW_INSTALL: self._test_msg_install, STATE_METER_INSTALL: self._test_msg_install, STATE_GROUP_INSTALL: self._test_msg_install, STATE_FLOW_EXIST_CHK: self._test_exist_check, STATE_THROUGHPUT_FLOW_EXIST_CHK: self._test_exist_check, STATE_METER_EXIST_CHK: self._test_exist_check, STATE_GROUP_EXIST_CHK: self._test_exist_check, STATE_TARGET_PKT_COUNT: self._test_get_packet_count, STATE_TESTER_PKT_COUNT: self._test_get_packet_count, STATE_FLOW_MATCH_CHK: self._test_flow_matching_check, STATE_NO_PKTIN_REASON: self._test_no_pktin_reason_check, STATE_GET_MATCH_COUNT: self._test_get_match_count, STATE_SEND_BARRIER: self._test_send_barrier, STATE_FLOW_UNMATCH_CHK: self._test_flow_unmatching_check, STATE_GET_THROUGHPUT: self._test_get_throughput, STATE_THROUGHPUT_CHK: self._test_throughput_check}
        self.send_msg_xids = []
        self.rcv_msgs = []
        self.state = state
        return test[state](*args)

    def _test_initialize_flow(self, datapath):
        xid = datapath.del_flows()
        self.send_msg_xids.append(xid)
        xid = datapath.add_flow(in_port=self.tester_recv_port_1, out_port=datapath.dp.ofproto.OFPP_CONTROLLER)
        self.send_msg_xids.append(xid)
        xid = datapath.send_barrier_request()
        self.send_msg_xids.append(xid)
        self._wait()
        assert len(self.rcv_msgs) == 1
        msg = self.rcv_msgs[0]
        assert isinstance(msg, datapath.dp.ofproto_parser.OFPBarrierReply)

    def _test_msg_install(self, datapath, message):
        xid = datapath.send_msg(message)
        self.send_msg_xids.append(xid)
        xid = datapath.send_barrier_request()
        self.send_msg_xids.append(xid)
        self._wait()
        assert len(self.rcv_msgs) == 1
        msg = self.rcv_msgs[0]
        assert isinstance(msg, datapath.dp.ofproto_parser.OFPBarrierReply)

    def _test_exist_check(self, method, message):
        ofp = method.__self__.dp.ofproto
        parser = method.__self__.dp.ofproto_parser
        method_dict = {OpenFlowSw.send_flow_stats.__name__: {'reply': parser.OFPFlowStatsReply, 'compare': self._compare_flow}}
        if ofp.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
            method_dict[OpenFlowSw.send_group_desc_stats.__name__] = {'reply': parser.OFPGroupDescStatsReply, 'compare': self._compare_group}
        if ofp.OFP_VERSION >= ofproto_v1_3.OFP_VERSION:
            method_dict[OpenFlowSw.send_meter_config_stats.__name__] = {'reply': parser.OFPMeterConfigStatsReply, 'compare': self._compare_meter}
        xid = method()
        self.send_msg_xids.append(xid)
        self._wait()
        ng_stats = []
        for msg in self.rcv_msgs:
            assert isinstance(msg, method_dict[method.__name__]['reply'])
            for stats in msg.body:
                result, stats = method_dict[method.__name__]['compare'](stats, message)
                if result:
                    return
                else:
                    ng_stats.append(stats)
        error_dict = {OpenFlowSw.send_flow_stats.__name__: {'flows': ', '.join(ng_stats)}}
        if ofp.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
            error_dict[OpenFlowSw.send_group_desc_stats.__name__] = {'groups': ', '.join(ng_stats)}
        if ofp.OFP_VERSION >= ofproto_v1_3.OFP_VERSION:
            error_dict[OpenFlowSw.send_meter_config_stats.__name__] = {'meters': ', '.join(ng_stats)}
        raise TestFailure(self.state, **error_dict[method.__name__])

    def _test_get_packet_count(self, is_target):
        sw = self.target_sw if is_target else self.tester_sw
        xid = sw.send_port_stats()
        self.send_msg_xids.append(xid)
        self._wait()
        result = {}
        for msg in self.rcv_msgs:
            for stats in msg.body:
                result[stats.port_no] = {'rx': stats.rx_packets, 'tx': stats.tx_packets}
        return result

    def _test_flow_matching_check(self, pkt):
        self.logger.debug('egress:[%s]', packet.Packet(pkt.get(KEY_EGRESS)))
        self.logger.debug('packet_in:[%s]', packet.Packet(pkt.get(KEY_PKT_IN)))
        try:
            self._wait()
        except TestTimeout:
            return TIMEOUT
        assert len(self.rcv_msgs) == 1
        msg = self.rcv_msgs[0]
        assert msg.__class__.__name__ == 'OFPPacketIn'
        self.logger.debug('dpid=%s : receive_packet[%s]', dpid_lib.dpid_to_str(msg.datapath.id), packet.Packet(msg.data))
        pkt_in_src_model = self.tester_sw if KEY_EGRESS in pkt else self.target_sw
        model_pkt = pkt[KEY_EGRESS] if KEY_EGRESS in pkt else pkt[KEY_PKT_IN]
        if hasattr(msg.datapath.ofproto, 'OFPR_NO_MATCH'):
            invalid_packet_in_reason = [msg.datapath.ofproto.OFPR_NO_MATCH]
        else:
            invalid_packet_in_reason = [msg.datapath.ofproto.OFPR_TABLE_MISS]
        if hasattr(msg.datapath.ofproto, 'OFPR_INVALID_TTL'):
            invalid_packet_in_reason.append(msg.datapath.ofproto.OFPR_INVALID_TTL)
        if msg.datapath.id != pkt_in_src_model.dp.id:
            pkt_type = 'packet-in'
            err_msg = 'SW[dpid=%s]' % dpid_lib.dpid_to_str(msg.datapath.id)
        elif msg.reason in invalid_packet_in_reason:
            pkt_type = 'packet-in'
            err_msg = 'OFPPacketIn[reason=%d]' % msg.reason
        elif repr(msg.data) != repr(model_pkt):
            pkt_type = 'packet'
            err_msg = self._diff_packets(packet.Packet(model_pkt), packet.Packet(msg.data))
        else:
            return TEST_OK
        raise TestFailure(self.state, pkt_type=pkt_type, detail=err_msg)

    def _test_no_pktin_reason_check(self, test_type, target_pkt_count, tester_pkt_count):
        before_target_receive = target_pkt_count[0][self.target_recv_port]['rx']
        before_target_send = target_pkt_count[0][self.target_send_port_1]['tx']
        before_tester_receive = tester_pkt_count[0][self.tester_recv_port_1]['rx']
        before_tester_send = tester_pkt_count[0][self.tester_send_port]['tx']
        after_target_receive = target_pkt_count[1][self.target_recv_port]['rx']
        after_target_send = target_pkt_count[1][self.target_send_port_1]['tx']
        after_tester_receive = tester_pkt_count[1][self.tester_recv_port_1]['rx']
        after_tester_send = tester_pkt_count[1][self.tester_send_port]['tx']
        if after_tester_send == before_tester_send:
            log_msg = 'no change in tx_packets on tester.'
        elif after_target_receive == before_target_receive:
            log_msg = 'no change in rx_packets on target.'
        elif test_type == KEY_EGRESS:
            if after_target_send == before_target_send:
                log_msg = 'no change in tx_packets on target.'
            elif after_tester_receive == before_tester_receive:
                log_msg = 'no change in rx_packets on tester.'
            else:
                log_msg = 'increment in rx_packets in tester.'
        else:
            assert test_type == KEY_PKT_IN
            log_msg = 'no packet-in.'
        raise TestFailure(self.state, detail=log_msg)

    def _test_get_match_count(self):
        xid = self.target_sw.send_table_stats()
        self.send_msg_xids.append(xid)
        self._wait()
        result = {}
        for msg in self.rcv_msgs:
            for stats in msg.body:
                result[stats.table_id] = {'lookup': stats.lookup_count, 'matched': stats.matched_count}
        return result

    def _test_send_barrier(self):
        xid = self.tester_sw.send_barrier_request()
        self.send_msg_xids.append(xid)
        self._wait()
        assert len(self.rcv_msgs) == 1
        msg = self.rcv_msgs[0]
        assert isinstance(msg, self.tester_sw.dp.ofproto_parser.OFPBarrierReply)

    def _test_flow_unmatching_check(self, before_stats, pkt):
        rcv_msgs = self._test_get_match_count()
        lookup = False
        for target_tbl_id in pkt[KEY_TBL_MISS]:
            before = before_stats[target_tbl_id]
            after = rcv_msgs[target_tbl_id]
            if before['lookup'] < after['lookup']:
                lookup = True
                if before['matched'] < after['matched']:
                    raise TestFailure(self.state)
        if not lookup:
            raise TestError(self.state)

    def _one_time_packet_send(self, pkt):
        self.logger.debug('send_packet:[%s]', packet.Packet(pkt[KEY_INGRESS]))
        xid = self.tester_sw.send_packet_out(pkt[KEY_INGRESS])
        self.send_msg_xids.append(xid)

    def _continuous_packet_send(self, pkt):
        assert self.ingress_event is None
        pkt_text = pkt[KEY_PACKETS]['packet_text']
        pkt_bin = pkt[KEY_PACKETS]['packet_binary']
        pktps = pkt[KEY_PACKETS][KEY_PKTPS]
        duration_time = pkt[KEY_PACKETS][KEY_DURATION_TIME]
        randomize = pkt[KEY_PACKETS]['randomize']
        self.logger.debug('send_packet:[%s]', packet.Packet(pkt_bin))
        self.logger.debug('pktps:[%d]', pktps)
        self.logger.debug('duration_time:[%d]', duration_time)
        arg = {'packet_text': pkt_text, 'packet_binary': pkt_bin, 'thread_counter': 0, 'dot_span': int(CONTINUOUS_PROGRESS_SPAN / CONTINUOUS_THREAD_INTVL), 'packet_counter': float(0), 'packet_counter_inc': pktps * CONTINUOUS_THREAD_INTVL, 'randomize': randomize}
        try:
            self.ingress_event = hub.Event()
            tid = hub.spawn(self._send_packet_thread, arg)
            self.ingress_threads.append(tid)
            self.ingress_event.wait(duration_time)
            if self.thread_msg is not None:
                raise self.thread_msg
        finally:
            sys.stdout.write('\r\n')
            sys.stdout.flush()

    def _send_packet_thread(self, arg):
        """ Send several packets continuously. """
        if self.ingress_event is None or self.ingress_event._cond:
            return
        if not arg['thread_counter'] % arg['dot_span']:
            sys.stdout.write('.')
            sys.stdout.flush()
        arg['thread_counter'] += 1
        arg['packet_counter'] += arg['packet_counter_inc']
        count = int(arg['packet_counter'])
        arg['packet_counter'] -= count
        hub.sleep(CONTINUOUS_THREAD_INTVL)
        tid = hub.spawn(self._send_packet_thread, arg)
        self.ingress_threads.append(tid)
        hub.sleep(0)
        for _ in range(count):
            if arg['randomize']:
                msg = eval('/'.join(arg['packet_text']))
                msg.serialize()
                data = msg.data
            else:
                data = arg['packet_binary']
            try:
                self.tester_sw.send_packet_out(data)
            except Exception as err:
                self.thread_msg = err
                self.ingress_event.set()
                break

    def _compare_flow(self, stats1, stats2):

        def __reasm_match(match):
            """ reassemble match_fields. """
            match_fields = match.to_jsondict()
            match_fields['OFPMatch'].pop('wildcards', None)
            return match_fields
        attr_list = ['cookie', 'priority', 'hard_timeout', 'idle_timeout', 'match']
        if self.target_sw.dp.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            attr_list += ['actions']
        else:
            attr_list += ['table_id', 'instructions']
        for attr in attr_list:
            value1 = getattr(stats1, attr)
            value2 = getattr(stats2, attr)
            if attr in ['actions', 'instructions']:
                value1 = sorted(value1, key=lambda x: x.type)
                value2 = sorted(value2, key=lambda x: x.type)
            elif attr == 'match':
                value1 = __reasm_match(value1)
                value2 = __reasm_match(value2)
            if str(value1) != str(value2):
                return (False, 'flow_stats(%s != %s)' % (value1, value2))
        return (True, None)

    @classmethod
    def _compare_meter(cls, stats1, stats2):
        """compare the message used to install and the message got from
           the switch."""
        attr_list = ['flags', 'meter_id', 'bands']
        for attr in attr_list:
            value1 = getattr(stats1, attr)
            value2 = getattr(stats2, attr)
            if str(value1) != str(value2):
                return (False, 'meter_stats(%s != %s)' % (value1, value2))
        return (True, None)

    @classmethod
    def _compare_group(cls, stats1, stats2):
        attr_list = ['type', 'group_id', 'buckets']
        for attr in attr_list:
            value1 = getattr(stats1, attr)
            value2 = getattr(stats2, attr)
            if str(value1) != str(value2):
                return (False, 'group_stats(%s != %s)' % (value1, value2))
            return (True, None)

    @classmethod
    def _diff_packets(cls, model_pkt, rcv_pkt):
        msg = []
        for rcv_p in rcv_pkt.protocols:
            if not isinstance(rcv_p, bytes):
                model_protocols = model_pkt.get_protocols(type(rcv_p))
                if len(model_protocols) == 1:
                    model_p = model_protocols[0]
                    diff = []
                    for attr in rcv_p.__dict__:
                        if attr.startswith('_'):
                            continue
                        if callable(attr):
                            continue
                        if hasattr(rcv_p.__class__, attr):
                            continue
                        rcv_attr = repr(getattr(rcv_p, attr))
                        model_attr = repr(getattr(model_p, attr))
                        if rcv_attr != model_attr:
                            diff.append('%s=%s' % (attr, rcv_attr))
                    if diff:
                        msg.append('%s(%s)' % (rcv_p.__class__.__name__, ','.join(diff)))
                elif not model_protocols or not str(rcv_p) in str(model_protocols):
                    msg.append(str(rcv_p))
            else:
                model_p = ''
                for p in model_pkt.protocols:
                    if isinstance(p, bytes):
                        model_p = p
                        break
                if model_p != rcv_p:
                    msg.append('str(%s)' % repr(rcv_p))
        if msg:
            return '/'.join(msg)
        else:
            return 'Encounter an error during packet comparison. it is malformed.'

    def _test_get_throughput(self):
        xid = self.tester_sw.send_flow_stats()
        self.send_msg_xids.append(xid)
        self._wait()
        assert len(self.rcv_msgs) == 1
        flow_stats = self.rcv_msgs[0].body
        self.logger.debug(flow_stats)
        result = {}
        for stat in flow_stats:
            if stat.cookie != THROUGHPUT_COOKIE:
                continue
            result[str(stat.match)] = (stat.byte_count, stat.packet_count)
        return (time.time(), result)

    def _test_throughput_check(self, throughputs, start, end):
        msgs = []
        elapsed_sec = end[0] - start[0]
        for throughput in throughputs:
            match = str(throughput[KEY_FLOW].match)
            fields = dict(throughput[KEY_FLOW].match._fields2)
            if match not in start[1] or match not in end[1]:
                raise TestError(self.state, match=match)
            increased_bytes = end[1][match][0] - start[1][match][0]
            increased_packets = end[1][match][1] - start[1][match][1]
            if throughput[KEY_PKTPS]:
                key = KEY_PKTPS
                conv = 1
                measured_value = increased_packets
                unit = 'pktps'
            elif throughput[KEY_KBPS]:
                key = KEY_KBPS
                conv = 1024 / 8
                measured_value = increased_bytes
                unit = 'kbps'
            else:
                raise OSKenException('An invalid key exists that is neither "%s" nor "%s".' % (KEY_KBPS, KEY_PKTPS))
            expected_value = throughput[key] * elapsed_sec * conv
            margin = expected_value * THROUGHPUT_THRESHOLD
            self.logger.debug('measured_value:[%s]', measured_value)
            self.logger.debug('expected_value:[%s]', expected_value)
            self.logger.debug('margin:[%s]', margin)
            if math.fabs(measured_value - expected_value) > margin:
                msgs.append('{0} {1:.2f}{2}'.format(fields, measured_value / elapsed_sec / conv, unit))
        if msgs:
            raise TestFailure(self.state, detail=', '.join(msgs))

    def _wait(self):
        """ Wait until specific OFP message received
             or timer is exceeded. """
        assert self.waiter is None
        self.waiter = hub.Event()
        self.rcv_msgs = []
        timeout = False
        timer = hub.Timeout(WAIT_TIMER)
        try:
            self.waiter.wait()
        except hub.Timeout as t:
            if t is not timer:
                raise OSKenException('Internal error. Not my timeout.')
            timeout = True
        finally:
            timer.cancel()
        self.waiter = None
        if timeout:
            raise TestTimeout(self.state)
        if self.rcv_msgs and isinstance(self.rcv_msgs[0], self.rcv_msgs[0].datapath.ofproto_parser.OFPErrorMsg):
            raise TestReceiveError(self.state, self.rcv_msgs[0])

    @set_ev_cls([ofp_event.EventOFPFlowStatsReply, ofp_event.EventOFPMeterConfigStatsReply, ofp_event.EventOFPTableStatsReply, ofp_event.EventOFPPortStatsReply, ofp_event.EventOFPGroupDescStatsReply], handler.MAIN_DISPATCHER)
    def stats_reply_handler(self, ev):
        ofp = ev.msg.datapath.ofproto
        event_states = {ofp_event.EventOFPFlowStatsReply: [STATE_FLOW_EXIST_CHK, STATE_THROUGHPUT_FLOW_EXIST_CHK, STATE_GET_THROUGHPUT], ofp_event.EventOFPTableStatsReply: [STATE_GET_MATCH_COUNT, STATE_FLOW_UNMATCH_CHK], ofp_event.EventOFPPortStatsReply: [STATE_TARGET_PKT_COUNT, STATE_TESTER_PKT_COUNT]}
        if ofp.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
            event_states[ofp_event.EventOFPGroupDescStatsReply] = [STATE_GROUP_EXIST_CHK]
        if ofp.OFP_VERSION >= ofproto_v1_3.OFP_VERSION:
            event_states[ofp_event.EventOFPMeterConfigStatsReply] = [STATE_METER_EXIST_CHK]
        if self.state in event_states[ev.__class__]:
            if self.waiter and ev.msg.xid in self.send_msg_xids:
                self.rcv_msgs.append(ev.msg)
                if not ev.msg.flags:
                    self.waiter.set()
                    hub.sleep(0)

    @set_ev_cls(ofp_event.EventOFPBarrierReply, handler.MAIN_DISPATCHER)
    def barrier_reply_handler(self, ev):
        state_list = [STATE_INIT_FLOW, STATE_INIT_THROUGHPUT_FLOW, STATE_INIT_METER, STATE_INIT_GROUP, STATE_FLOW_INSTALL, STATE_THROUGHPUT_FLOW_INSTALL, STATE_METER_INSTALL, STATE_GROUP_INSTALL, STATE_SEND_BARRIER]
        if self.state in state_list:
            if self.waiter and ev.msg.xid in self.send_msg_xids:
                self.rcv_msgs.append(ev.msg)
                self.waiter.set()
                hub.sleep(0)

    @set_ev_cls(ofp_event.EventOFPPacketIn, handler.MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        state_list = [STATE_FLOW_MATCH_CHK]
        if self.state in state_list:
            if self.waiter:
                self.rcv_msgs.append(ev.msg)
                self.waiter.set()
                hub.sleep(0)

    @set_ev_cls(ofp_event.EventOFPErrorMsg, [handler.HANDSHAKE_DISPATCHER, handler.CONFIG_DISPATCHER, handler.MAIN_DISPATCHER])
    def error_msg_handler(self, ev):
        if ev.msg.xid in self.send_msg_xids:
            self.rcv_msgs.append(ev.msg)
            if self.waiter:
                self.waiter.set()
                hub.sleep(0)