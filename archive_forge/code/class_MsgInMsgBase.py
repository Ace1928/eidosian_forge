import base64
import collections
import logging
import struct
import functools
from os_ken import exception
from os_ken import utils
from os_ken.lib import stringify
from os_ken.ofproto import ofproto_common
class MsgInMsgBase(MsgBase):

    @classmethod
    def _decode_value(cls, k, json_value, decode_string=base64.b64decode, **additional_args):
        return cls._get_decoder(k, decode_string)(json_value, **additional_args)