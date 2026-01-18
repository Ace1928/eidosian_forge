from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def _prepare_route_params(self, input_dict):
    result = {}
    route = self.find_route(input_dict)
    for url_param in route.params_in_path:
        if url_param in input_dict:
            result[url_param] = input_dict[url_param]
    return result