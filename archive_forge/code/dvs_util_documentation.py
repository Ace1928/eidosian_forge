import logging
from oslo_vmware import vim_util
Delete a specific port group

    :param session: vCenter soap session
    :param portgroup_moref: managed portgroup object reference
    