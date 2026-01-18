import sys
import dns._features
def _config_nameservers(self, nameservers):
    split_char = self._determine_split_char(nameservers)
    ns_list = nameservers.split(split_char)
    for ns in ns_list:
        if ns not in self.info.nameservers:
            self.info.nameservers.append(ns)