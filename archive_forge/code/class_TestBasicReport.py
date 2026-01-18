from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
class TestBasicReport(base.BaseTestCase):

    def setUp(self):
        super(TestBasicReport, self).setUp()
        self.report = report.BasicReport()

    def test_add_section(self):
        self.report.add_section(BasicView(), basic_generator)
        self.assertEqual(len(self.report.sections), 1)

    def test_append_section(self):
        self.report.add_section(BasicView(), lambda: {'a': 1})
        self.report.add_section(BasicView(), basic_generator)
        self.assertEqual(len(self.report.sections), 2)
        self.assertEqual(self.report.sections[1].generator, basic_generator)

    def test_insert_section(self):
        self.report.add_section(BasicView(), lambda: {'a': 1})
        self.report.add_section(BasicView(), basic_generator, 0)
        self.assertEqual(len(self.report.sections), 2)
        self.assertEqual(self.report.sections[0].generator, basic_generator)

    def test_basic_render(self):
        self.report.add_section(BasicView(), basic_generator)
        self.assertEqual(self.report.run(), 'int: 1;string: value;')