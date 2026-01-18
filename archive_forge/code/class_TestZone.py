from boto.route53.zone import Zone
from tests.compat import mock, unittest
class TestZone(unittest.TestCase):

    def test_find_records(self):
        mock_connection = mock.Mock()
        zone = Zone(mock_connection, {})
        zone.id = None
        rr_names = ['amazon.com', 'amazon.com', 'aws.amazon.com', 'aws.amazon.com']
        mock_rrs = []
        for rr_name in rr_names:
            mock_rr = mock.Mock()
            mock_rr.name = rr_name
            mock_rr.type = 'A'
            mock_rr.weight = None
            mock_rr.region = None
            mock_rrs.append(mock_rr)
        mock_rrs[3] = None
        mock_connection.get_all_rrsets.return_value = mock_rrs
        mock_connection._make_qualified.return_value = 'amazon.com'
        try:
            result_rrs = zone.find_records('amazon.com', 'A', all=True)
        except AttributeError as e:
            self.fail('find_records() iterated too far into resource record list.')
        self.assertEqual(result_rrs, [mock_rrs[0], mock_rrs[1]])