import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
def _pretty_list(self, func, *args, **kwargs):
    result = self._safe_exec(func, *args, **kwargs)
    if self.verbose:
        return
    if result and len(result) > 0:
        for item in result:
            print(self._dumps(item._info))
    else:
        print('OK')