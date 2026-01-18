from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddSystemConfigFlag(parser, hidden=True):
    """Adds --system-config-from-file flag to the given parser."""
    parser.add_argument('--system-config-from-file', type=arg_parsers.FileContents(), hidden=hidden, help="\nPath of the YAML/JSON file that contains the node configuration, including\nLinux kernel parameters (sysctls) and kubelet configs.\n\nExamples:\n\n    kubeletConfig:\n      cpuManagerPolicy: static\n    linuxConfig:\n      sysctl:\n        net.core.somaxconn: '2048'\n        net.ipv4.tcp_rmem: '4096 87380 6291456'\n    hugepageConfig:\n      hugepage_size2m: '1024'\n      hugepage_size1g: '2'\n\nList of supported kubelet configs in 'kubeletConfig'.\n\nKEY                                 | VALUE\n----------------------------------- | ----------------------------------\ncpuManagerPolicy                    | either 'static' or 'none'\ncpuCFSQuota                         | true or false (enabled by default)\ncpuCFSQuotaPeriod                   | interval (e.g., '100ms')\npodPidsLimit                        | integer (The value must be greater than or equal to 1024 and less than 4194304.)\n\nList of supported sysctls in 'linuxConfig'.\n\nKEY                                        | VALUE\n------------------------------------------ | ------------------------------------------\nnet.core.netdev_max_backlog                | Any positive integer, less than 2147483647\nnet.core.rmem_max                          | Any positive integer, less than 2147483647\nnet.core.wmem_default                      | Any positive integer, less than 2147483647\nnet.core.wmem_max                          | Any positive integer, less than 2147483647\nnet.core.optmem_max                        | Any positive integer, less than 2147483647\nnet.core.somaxconn                         | Must be [128, 2147483647]\nnet.ipv4.tcp_rmem                          | Any positive integer tuple\nnet.ipv4.tcp_wmem                          | Any positive integer tuple\nnet.ipv4.tcp_tw_reuse                      | Must be {0, 1}\n\nList of supported hugepage size in 'hugepageConfig'.\n\nKEY             | VALUE\n----------------| ---------------------------------------------\nhugepage_size2m | Number of 2M huge pages, any positive integer\nhugepage_size1g | Number of 1G huge pages, any positive integer\n\nAllocated hugepage size should not exceed 60% of available memory on the node. For example, c2d-highcpu-4 has 8GB memory, total\nallocated hugepage of 2m and 1g should not exceed 8GB * 0.6 = 4.8GB.\n\n1G hugepages are only available in following machine familes:\nc3, m2, c2d, c3d, h3, m3, a2, a3, g2.\n\nNote, updating the system configuration of an existing node pool requires recreation of the nodes which which might cause a disruption.\n")