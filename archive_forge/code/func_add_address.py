import six
from pycadf import attachment
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import geolocation
from pycadf import host
from pycadf import identifier
def add_address(self, addr):
    """Add CADF endpoints to Resource

        :param addr: CADF Endpoint to add to Resource
        """
    if addr is not None and isinstance(addr, endpoint.Endpoint):
        if addr.is_valid():
            if not hasattr(self, RESOURCE_KEYNAME_ADDRS):
                setattr(self, RESOURCE_KEYNAME_ADDRS, list())
            addrs = getattr(self, RESOURCE_KEYNAME_ADDRS)
            addrs.append(addr)
        else:
            raise ValueError('Invalid endpoint')
    else:
        raise ValueError('Invalid endpoint. Value must be an Endpoint')