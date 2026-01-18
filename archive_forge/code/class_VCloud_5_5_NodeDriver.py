import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class VCloud_5_5_NodeDriver(VCloud_5_1_NodeDriver):
    """Use 5.5 Connection class to explicitly set 5.5 for the version in
    Accept headers
    """
    connectionCls = VCloud_5_5_Connection

    def ex_create_snapshot(self, node):
        """
        Creates new snapshot of a virtual machine or of all
        the virtual machines in a vApp. Prior to creation of the new
        snapshots, any existing user created snapshots associated
        with the virtual machines are removed.

        :param  node: node
        :type   node: :class:`Node`

        :rtype: :class:`Node`
        """
        snapshot_xml = ET.Element('CreateSnapshotParams', {'memory': 'true', 'name': 'name', 'quiesce': 'true', 'xmlns': 'http://www.vmware.com/vcloud/v1.5', 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})
        ET.SubElement(snapshot_xml, 'Description').text = 'Description'
        content_type = 'application/vnd.vmware.vcloud.createSnapshotParams+xml'
        headers = {'Content-Type': content_type}
        return self._perform_snapshot_operation(node, 'createSnapshot', snapshot_xml, headers)

    def ex_remove_snapshots(self, node):
        """
        Removes all user created snapshots for a vApp or virtual machine.

        :param  node: node
        :type   node: :class:`Node`

        :rtype: :class:`Node`
        """
        return self._perform_snapshot_operation(node, 'removeAllSnapshots', None, None)

    def ex_revert_to_snapshot(self, node):
        """
        Reverts a vApp or virtual machine to the current snapshot, if any.

        :param  node: node
        :type   node: :class:`Node`

        :rtype: :class:`Node`
        """
        return self._perform_snapshot_operation(node, 'revertToCurrentSnapshot', None, None)

    def _perform_snapshot_operation(self, node, operation, xml_data, headers):
        res = self.connection.request('{}/action/{}'.format(get_url_path(node.id), operation), data=ET.tostring(xml_data) if xml_data is not None else None, method='POST', headers=headers)
        self._wait_for_task_completion(res.object.get('href'))
        res = self.connection.request(get_url_path(node.id))
        return self._to_node(res.object)

    def ex_acquire_mks_ticket(self, vapp_or_vm_id, vm_num=0):
        """
        Retrieve a mks ticket that you can use to gain access to the console
        of a running VM. If successful, returns a dict with the following
        keys:

          - host: host (or proxy) through which the console connection
                is made
          - vmx: a reference to the VMX file of the VM for which this
               ticket was issued
          - ticket: screen ticket to use to authenticate the client
          - port: host port to be used for console access

        :param  vapp_or_vm_id: vApp or VM ID you want to connect to.
        :type   vapp_or_vm_id: ``str``

        :param  vm_num: If a vApp ID is provided, vm_num is position in the
                vApp VM list of the VM you want to get a screen ticket.
                Default is 0.
        :type   vm_num: ``int``

        :rtype: ``dict``
        """
        vm = self._get_vm_elements(vapp_or_vm_id)[vm_num]
        try:
            res = self.connection.request('%s/screen/action/acquireMksTicket' % get_url_path(vm.get('href')), method='POST')
            output = {'host': res.object.find(fixxpath(res.object, 'Host')).text, 'vmx': res.object.find(fixxpath(res.object, 'Vmx')).text, 'ticket': res.object.find(fixxpath(res.object, 'Ticket')).text, 'port': res.object.find(fixxpath(res.object, 'Port')).text}
            return output
        except Exception:
            return None