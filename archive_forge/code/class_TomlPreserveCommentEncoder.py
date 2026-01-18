import datetime
import re
import sys
from decimal import Decimal
from toml.decoder import InlineTableDict
class TomlPreserveCommentEncoder(TomlEncoder):

    def __init__(self, _dict=dict, preserve=False):
        from toml.decoder import CommentValue
        super(TomlPreserveCommentEncoder, self).__init__(_dict, preserve)
        self.dump_funcs[CommentValue] = lambda v: v.dump(self.dump_value)