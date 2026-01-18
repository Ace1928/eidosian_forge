from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataServerCpuSpecification:
    """
    A class that represents the specification of the CPU(s) for a
    node
    """

    def __init__(self, cpu_count, cores_per_socket, performance):
        """
        Instantiate a new :class:`DimensionDataServerCpuSpecification`

        :param cpu_count: The number of CPUs
        :type  cpu_count: ``int``

        :param cores_per_socket: The number of cores per socket, the
            recommendation is 1
        :type  cores_per_socket: ``int``

        :param performance: The performance type, e.g. HIGHPERFORMANCE
        :type  performance: ``str``
        """
        self.cpu_count = cpu_count
        self.cores_per_socket = cores_per_socket
        self.performance = performance

    def __repr__(self):
        return '<DimensionDataServerCpuSpecification: cpu_count=%s, cores_per_socket=%s, performance=%s>' % (self.cpu_count, self.cores_per_socket, self.performance)