import sys
from collections import namedtuple
class LazyLoadedWaiterModel(object):
    """A lazily loaded waiter model

    This does not load the service waiter model until an attempt is made
    to retrieve the waiter model for a specific waiter. This is helpful
    in docstring generation where we do not need to actually need to grab
    the waiter-2.json until it is accessed through a ``get_waiter`` call
    when the docstring is generated/accessed.
    """

    def __init__(self, bc_session, service_name, api_version):
        self._session = bc_session
        self._service_name = service_name
        self._api_version = api_version

    def get_waiter(self, waiter_name):
        return self._session.get_waiter_model(self._service_name, self._api_version).get_waiter(waiter_name)