from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
def _test_get_element_associated_class(self, fields=None):
    mock_conn = mock.MagicMock()
    _wqlutils.get_element_associated_class(mock_conn, mock.sentinel.class_name, element_instance_id=mock.sentinel.instance_id, fields=fields)
    expected_fields = ', '.join(fields) if fields else '*'
    expected_query = "SELECT %(expected_fields)s FROM %(class_name)s WHERE InstanceID LIKE '%(instance_id)s%%'" % {'expected_fields': expected_fields, 'class_name': mock.sentinel.class_name, 'instance_id': mock.sentinel.instance_id}
    mock_conn.query.assert_called_once_with(expected_query)