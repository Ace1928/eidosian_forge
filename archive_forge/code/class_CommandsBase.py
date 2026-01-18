import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
class CommandsBase(object):
    params = []

    def __init__(self, parser):
        self._parse_options(parser)

    def _get_client(self):
        """Creates the all important client object."""
        try:
            client_cls = client.TroveHTTPClient
            if self.verbose:
                client.log_to_streamhandler(sys.stdout)
                client.RDC_PP = True
            return client.Dbaas(self.username, self.apikey, self.tenant_id, auth_url=self.auth_url, auth_strategy=self.auth_type, service_type=self.service_type, service_name=self.service_name, region_name=self.region, service_url=self.service_url, insecure=self.insecure, client_cls=client_cls)
        except Exception:
            if self.debug:
                raise
            print(sys.exc_info()[1])

    def _safe_exec(self, func, *args, **kwargs):
        if not self.debug:
            try:
                return func(*args, **kwargs)
            except Exception:
                print(sys.exc_info()[1])
                return None
        else:
            return func(*args, **kwargs)

    @classmethod
    def _prepare_parser(cls, parser):
        for param in cls.params:
            parser.add_option('--%s' % param)

    def _parse_options(self, parser):
        opts, args = parser.parse_args()
        for param in opts.__dict__:
            value = getattr(opts, param)
            setattr(self, param, value)

    def _require(self, *params):
        for param in params:
            if not hasattr(self, param):
                raise ArgumentRequired(param)
            if not getattr(self, param):
                raise ArgumentRequired(param)

    def _require_at_least_one_of(self, *params):
        argument_present = False
        for param in params:
            if hasattr(self, param):
                if getattr(self, param):
                    argument_present = True
        if argument_present is False:
            raise ArgumentsRequired(*params)

    def _make_list(self, *params):
        for param in params:
            raw = getattr(self, param)
            if isinstance(raw, list):
                return
            raw = [item.strip() for item in raw.split(',')]
            setattr(self, param, raw)

    def _pretty_print(self, func, *args, **kwargs):
        if self.verbose:
            self._safe_exec(func, *args, **kwargs)
            return

        def wrapped_func():
            result = func(*args, **kwargs)
            if result:
                print(json.dumps(result._info, sort_keys=True, indent=4))
            else:
                print('OK')
        self._safe_exec(wrapped_func)

    def _dumps(self, item):
        return json.dumps(item, sort_keys=True, indent=4)

    def _pretty_list(self, func, *args, **kwargs):
        result = self._safe_exec(func, *args, **kwargs)
        if self.verbose:
            return
        if result and len(result) > 0:
            for item in result:
                print(self._dumps(item._info))
        else:
            print('OK')

    def _pretty_paged(self, func, *args, **kwargs):
        try:
            limit = self.limit
            if limit:
                limit = int(limit, 10)
            result = func(*args, limit=limit, marker=self.marker, **kwargs)
            if self.verbose:
                return
            if result and len(result) > 0:
                for item in result:
                    print(self._dumps(item._info))
                if result.links:
                    print('Links:')
                    for link in result.links:
                        print(self._dumps(link))
            else:
                print('OK')
        except Exception:
            if self.debug:
                raise
            print(sys.exc_info()[1])