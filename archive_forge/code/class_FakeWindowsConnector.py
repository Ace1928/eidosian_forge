from os_brick.initiator.windows import base as win_conn_base
class FakeWindowsConnector(win_conn_base.BaseWindowsConnector):

    def connect_volume(self, connection_properties):
        return {}

    def disconnect_volume(self, connection_properties, device_info, force=False, ignore_errors=False):
        pass

    def get_volume_paths(self, connection_properties):
        return []

    def get_search_path(self):
        return None

    def get_all_available_volumes(self, connection_properties=None):
        return []