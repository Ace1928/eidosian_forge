from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_ERROR
from os_ken.services.protocols.bgp.operator.internal_api import WrongParamError
class WrongParamResp(object):

    def __new__(cls, e=None):
        return cls.wrong_param_resp_factory(e)

    @staticmethod
    def wrong_param_resp_factory(e=None):
        if not e:
            e = WrongParamError()
        desc = 'wrong parameters: %s' % str(e)
        return CommandsResponse(STATUS_ERROR, desc)