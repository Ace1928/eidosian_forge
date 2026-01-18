from magnumclient.common import base
from magnumclient.common import utils
from magnumclient import exceptions
Retrieve a list of clusters.

        :param marker: Optional, the UUID of a cluster, eg the last
                       cluster from a previous result set. Return
                       the next result set.
        :param limit: The maximum number of results to return per
                      request, if:

            1) limit > 0, the maximum number of clusters to return.
            2) limit == 0, return the entire list of clusters.
            3) limit param is NOT specified (None), the number of items
               returned respect the maximum imposed by the Magnum API
               (see Magnum's api.max_limit option).

        :param sort_key: Optional, field used for sorting.

        :param sort_dir: Optional, direction of sorting, either 'asc' (the
                         default) or 'desc'.

        :param detail: Optional, boolean whether to return detailed information
                       about clusters.

        :returns: A list of clusters.

        