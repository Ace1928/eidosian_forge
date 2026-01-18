import datetime
import keystone.tests.unit as tests
class TestTimeUtils(tests.BaseTestCase):

    def test_parsing_date_strings_returns_a_datetime(self):
        example_date_str = '2015-09-23T04:45:37.196621Z'
        dt = datetime.datetime.strptime(example_date_str, tests.TIME_FORMAT)
        self.assertIsInstance(dt, datetime.datetime)

    def test_parsing_invalid_date_strings_raises_a_ValueError(self):
        example_date_str = ''
        simple_format = '%Y'
        self.assertRaises(ValueError, datetime.datetime.strptime, example_date_str, simple_format)