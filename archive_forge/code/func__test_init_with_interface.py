import testtools
from unittest import mock
from keystoneauth1.exceptions import catalog
from magnumclient.v1 import client
def _test_init_with_interface(self, init_func, mock_load_service_type, mock_load_session, mock_http_client):
    expected_interface = 'admin'
    session = mock.Mock()
    mock_load_session.return_value = session
    init_func(expected_interface)
    mock_load_session.assert_called_once_with(**self._load_session_kwargs())
    expected_kwargs = self._load_service_type_kwargs()
    expected_kwargs['interface'] = expected_interface
    mock_load_service_type.assert_called_once_with(session, **expected_kwargs)
    expected_kwargs = self._session_client_kwargs(session)
    expected_kwargs['interface'] = expected_interface
    mock_http_client.assert_called_once_with(**expected_kwargs)