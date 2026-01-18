import math
import enum
from pyomo.opt.results.container import MapContainer, ListContainer, ignore
from pyomo.common.collections import Bunch, OrderedDict
class SolutionSet(ListContainer):

    def __init__(self):
        ListContainer.__init__(self, Solution)
        self._option = default_print_options

    def _repn_(self, option):
        if not option.schema and (not self._active) and (not self._required):
            return ignore
        if option.schema and len(self) == 0:
            self.add()
            self.add()
        if option.num_solutions is None:
            num = len(self)
        else:
            num = min(option.num_solutions, len(self))
        i = 0
        tmp = []
        for item in self._list:
            tmp.append(item._repn_(option))
            i = i + 1
            if i == num:
                break
        return [OrderedDict([('number of solutions', len(self)), ('number of solutions displayed', num)])] + tmp

    def __len__(self):
        return len(self._list)

    def __call__(self, i=1):
        return self._list[i - 1]

    def pprint(self, ostream, option, prefix='', repn=None):
        if not option.schema and (not self._active) and (not self._required):
            return ignore
        ostream.write('\n')
        ostream.write(prefix + '- ')
        spaces = ''
        for key in repn[0]:
            ostream.write(prefix + spaces + key + ': ' + str(repn[0][key]) + '\n')
            spaces = '  '
        i = 0
        for i in range(len(self._list)):
            item = self._list[i]
            ostream.write(prefix + '- ')
            item.pprint(ostream, option, from_list=True, prefix=prefix + '  ', repn=repn[i + 1])

    def load(self, repn):
        for data in repn[1:]:
            item = self.add()
            item.load(data)