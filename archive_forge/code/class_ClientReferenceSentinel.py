from contextlib import contextmanager
from abc import ABC
from abc import abstractmethod
class ClientReferenceSentinel(ABC):

    def __init__(self, client_id, id):
        self.client_id = client_id
        self.id = id

    def __reduce__(self):
        remote_obj = self.get_remote_obj()
        if remote_obj is None:
            return (self.__class__, (self.client_id, self.id))
        return (identity, (remote_obj,))

    @abstractmethod
    def get_remote_obj(self):
        pass

    def get_real_ref_from_server(self):
        global _current_server
        if _current_server is None:
            return None
        client_map = _current_server.client_side_ref_map.get(self.client_id, None)
        if client_map is None:
            return None
        return client_map.get(self.id, None)