import datetime
import io
import os
import re
import signal
import sys
import threading
from unittest import mock
import fixtures
import greenlet
from oslotest import base
import oslo_config
from oslo_config import fixture
from oslo_reports import guru_meditation_report as gmr
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import opts
class TestGuruMeditationReport(base.BaseTestCase):

    def setUp(self):
        super(TestGuruMeditationReport, self).setUp()
        self.curr_g = greenlet.getcurrent()
        self.report = gmr.TextGuruMeditation(FakeVersionObj())
        self.old_stderr = None
        self.CONF = self.useFixture(GmrConfigFixture(CONF)).conf

    def test_basic_report(self):
        report_lines = self.report.run().split('\n')
        target_str_header = ['========================================================================', '====                        Guru Meditation                         ====', '========================================================================', '||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||', '', '', '========================================================================', '====                            Package                             ====', '========================================================================', 'product = Sharp Cheddar', 'vendor = Cheese Shoppe', 'version = 1.0.0', '========================================================================', '====                            Threads                             ====', '========================================================================']
        self.assertEqual(target_str_header, report_lines[0:len(target_str_header)])
        self.assertTrue(re.match('------(\\s+)Thread #-?\\d+\\1\\s?------', report_lines[len(target_str_header)]))
        self.assertEqual('', report_lines[len(target_str_header) + 1])
        curr_line = skip_body_lines(len(target_str_header) + 2, report_lines)
        target_str_gt = ['========================================================================', '====                         Green Threads                          ====', '========================================================================', '------                        Green Thread                        ------', '']
        end_bound = curr_line + len(target_str_gt)
        self.assertEqual(target_str_gt, report_lines[curr_line:end_bound])
        curr_line = skip_body_lines(curr_line + len(target_str_gt), report_lines)
        target_str_p_head = ['========================================================================', '====                           Processes                            ====', '========================================================================']
        end_bound = curr_line + len(target_str_p_head)
        self.assertEqual(target_str_p_head, report_lines[curr_line:end_bound])
        curr_line += len(target_str_p_head)
        self.assertTrue(re.match('Process \\d+ \\(under \\d+\\)', report_lines[curr_line]))
        curr_line = skip_body_lines(curr_line + 1, report_lines)
        target_str_config = ['========================================================================', '====                         Configuration                          ====', '========================================================================', '']
        end_bound = curr_line + len(target_str_config)
        self.assertEqual(target_str_config, report_lines[curr_line:end_bound])

    def test_reg_persistent_section(self):

        def fake_gen():
            fake_data = {'cheddar': ['sharp', 'mild'], 'swiss': ['with holes', 'with lots of holes'], 'american': ['orange', 'yellow']}
            return mwdv.ModelWithDefaultViews(data=fake_data)
        gmr.TextGuruMeditation.register_section('Cheese Types', fake_gen)
        report_lines = self.report.run()
        target_lst = ['========================================================================', '====                          Cheese Types                          ====', '========================================================================', 'american = ', '  orange', '  yellow', 'cheddar = ', '  mild', '  sharp', 'swiss = ', '  with holes', '  with lots of holes']
        target_str = '\n'.join(target_lst)
        self.assertIn(target_str, report_lines)

    def test_register_autorun(self):
        gmr.TextGuruMeditation.setup_autorun(FakeVersionObj())
        self.old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        os.kill(os.getpid(), signal.SIGUSR2)
        self.assertIn('Guru Meditation', sys.stderr.getvalue())

    @mock.patch.object(gmr.TextGuruMeditation, '_setup_file_watcher')
    def test_register_autorun_without_signals(self, mock_setup_fh):
        version = FakeVersionObj()
        gmr.TextGuruMeditation.setup_autorun(version, conf=self.CONF)
        mock_setup_fh.assert_called_once_with('/specific/file', 10, version, None, '/var/fake_log')

    @mock.patch('os.stat')
    @mock.patch('time.sleep')
    @mock.patch.object(threading.Thread, 'start')
    def test_setup_file_watcher(self, mock_thread, mock_sleep, mock_stat):
        version = FakeVersionObj()
        mock_stat.return_value.st_mtime = 3
        gmr.TextGuruMeditation._setup_file_watcher(self.CONF.oslo_reports.file_event_handler, self.CONF.oslo_reports.file_event_handler_interval, version, None, self.CONF.oslo_reports.log_dir)
        mock_stat.assert_called_once_with('/specific/file')
        self.assertEqual(1, mock_thread.called)

    @mock.patch('oslo_utils.timeutils.utcnow', return_value=datetime.datetime(2014, 1, 1, 12, 0, 0))
    def test_register_autorun_log_dir(self, mock_strtime):
        log_dir = self.useFixture(fixtures.TempDir()).path
        gmr.TextGuruMeditation.setup_autorun(FakeVersionObj(), 'fake-service', log_dir)
        os.kill(os.getpid(), signal.SIGUSR2)
        with open(os.path.join(log_dir, 'fake-service_gurumeditation_20140101120000')) as df:
            self.assertIn('Guru Meditation', df.read())

    @mock.patch.object(gmr.TextGuruMeditation, 'run')
    def test_fail_prints_traceback(self, run_mock):

        class RunFail(Exception):
            pass
        run_mock.side_effect = RunFail()
        gmr.TextGuruMeditation.setup_autorun(FakeVersionObj())
        self.old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        os.kill(os.getpid(), signal.SIGUSR2)
        self.assertIn('RunFail', sys.stderr.getvalue())

    def tearDown(self):
        super(TestGuruMeditationReport, self).tearDown()
        if self.old_stderr is not None:
            sys.stderr = self.old_stderr