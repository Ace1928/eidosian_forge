import logging
import threading
import confluent_kafka
from confluent_kafka import KafkaException
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers.kafka_driver import kafka_options
class ConsumerConnection(Connection):
    """This is the class for kafka topic/assigned partition consumer
    """

    def __init__(self, conf, url):
        super(ConsumerConnection, self).__init__(conf, url)
        self.consumer = None
        self.consumer_timeout = self.driver_conf.kafka_consumer_timeout
        self.max_fetch_bytes = self.driver_conf.kafka_max_fetch_bytes
        self.group_id = self.driver_conf.consumer_group
        self.use_auto_commit = self.driver_conf.enable_auto_commit
        self.max_poll_records = self.driver_conf.max_poll_records
        self._consume_loop_stopped = False
        self.assignment_dict = dict()

    def find_assignment(self, topic, partition):
        """Find and return existing assignment based on topic and partition"""
        skey = '%s %d' % (topic, partition)
        return self.assignment_dict.get(skey)

    def on_assign(self, consumer, topic_partitions):
        """Rebalance on_assign callback"""
        assignment = [AssignedPartition(p.topic, p.partition) for p in topic_partitions]
        self.assignment_dict = {a.skey: a for a in assignment}
        for t in topic_partitions:
            LOG.debug('Topic %s assigned to partition %d', t.topic, t.partition)

    def on_revoke(self, consumer, topic_partitions):
        """Rebalance on_revoke callback"""
        self.assignment_dict = dict()
        for t in topic_partitions:
            LOG.debug('Topic %s revoked from partition %d', t.topic, t.partition)

    def _poll_messages(self, timeout):
        """Consume messages, callbacks and return list of messages"""
        msglist = self.consumer.consume(self.max_poll_records, timeout)
        if len(self.assignment_dict) == 0 or len(msglist) == 0:
            raise ConsumerTimeout()
        messages = []
        for message in msglist:
            if message is None:
                break
            a = self.find_assignment(message.topic(), message.partition())
            if a is None:
                LOG.warning('Message for %s received on unassigned partition %d', message.topic(), message.partition())
            else:
                messages.append(message.value())
        if not self.use_auto_commit:
            self.consumer.commit(asynchronous=False)
        return messages

    def consume(self, timeout=None):
        """Receive messages.

        :param timeout: poll timeout in seconds
        """

        def _raise_timeout(exc):
            raise driver_common.Timeout(str(exc))
        timer = driver_common.DecayingTimer(duration=timeout)
        timer.start()
        poll_timeout = self.consumer_timeout if timeout is None else min(timeout, self.consumer_timeout)
        while True:
            if self._consume_loop_stopped:
                return
            try:
                if eventletutils.is_monkey_patched('thread'):
                    return tpool.execute(self._poll_messages, poll_timeout)
                return self._poll_messages(poll_timeout)
            except ConsumerTimeout as exc:
                poll_timeout = timer.check_return(_raise_timeout, exc, maximum=self.consumer_timeout)
            except Exception:
                LOG.exception('Failed to consume messages')
                return

    def stop_consuming(self):
        self._consume_loop_stopped = True

    def close(self):
        if self.consumer:
            self.consumer.close()
            self.consumer = None

    def declare_topic_consumer(self, topics, group=None):
        conf = {'bootstrap.servers': ','.join(self.hostaddrs), 'group.id': group or self.group_id, 'enable.auto.commit': self.use_auto_commit, 'max.partition.fetch.bytes': self.max_fetch_bytes, 'security.protocol': self.security_protocol, 'sasl.mechanism': self.sasl_mechanism, 'sasl.username': self.username, 'sasl.password': self.password, 'ssl.ca.location': self.ssl_cafile, 'ssl.certificate.location': self.ssl_client_cert_file, 'ssl.key.location': self.ssl_client_key_file, 'ssl.key.password': self.ssl_client_key_password, 'enable.partition.eof': False, 'default.topic.config': {'auto.offset.reset': 'latest'}}
        LOG.debug('Subscribing to %s as %s', topics, group or self.group_id)
        self.consumer = confluent_kafka.Consumer(conf)
        self.consumer.subscribe(topics, on_assign=self.on_assign, on_revoke=self.on_revoke)