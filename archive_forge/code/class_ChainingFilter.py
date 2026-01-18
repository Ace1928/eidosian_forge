import os
import re
import shutil
import sys
class ChainingFilter(CommandFilter):

    def exec_args(self, userargs):
        return []