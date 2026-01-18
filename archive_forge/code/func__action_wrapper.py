from collections import namedtuple
import json
import logging
import pprint
import re
def _action_wrapper(self, params):
    filter_params = []
    if '|' in params:
        ind = params.index('|')
        new_params = params[:ind]
        filter_params = params[ind:]
        params = new_params
    action_resp = self.action(params)
    if len(filter_params) > 1:
        action_resp = self.filter_resp(action_resp, filter_params[1:])
    action_resp = CommandsResponse(action_resp.status, self.resp_formatter(action_resp))
    return (action_resp, self.__class__)