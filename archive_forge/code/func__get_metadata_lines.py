import collections
from llvmlite.ir import context, values, types, _utils
def _get_metadata_lines(self):
    mdbuf = []
    for k, v in self.namedmetadata.items():
        mdbuf.append('!{name} = !{{ {operands} }}'.format(name=k, operands=', '.join((i.get_reference() for i in v.operands))))
    for md in self.metadata:
        mdbuf.append(str(md))
    return mdbuf