from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.influxdb import InfluxDb
class AnsibleInfluxDBRead(InfluxDb):

    def read_by_query(self, query):
        client = self.connect_to_influxdb()
        try:
            rs = client.query(query)
            if rs:
                return list(rs.get_points())
        except Exception as e:
            self.module.fail_json(msg=to_native(e))