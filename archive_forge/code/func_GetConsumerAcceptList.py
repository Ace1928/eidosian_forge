from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
def GetConsumerAcceptList(args, messages):
    """Get consumer accept list of the service attachment."""
    consumer_accept_list = []
    for project_limit in args.consumer_accept_list:
        for project_id_or_network_url, conn_limit in sorted(six.iteritems(project_limit)):
            if '/networks/' in project_id_or_network_url:
                consumer_accept_list.append(messages.ServiceAttachmentConsumerProjectLimit(networkUrl=project_id_or_network_url, connectionLimit=int(conn_limit)))
            else:
                consumer_accept_list.append(messages.ServiceAttachmentConsumerProjectLimit(projectIdOrNum=project_id_or_network_url, connectionLimit=int(conn_limit)))
    return consumer_accept_list