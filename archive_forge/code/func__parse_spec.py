import re
from sphinx.locale import _
from sphinx.ext.napoleon.docstring import NumpyDocstring
def _parse_spec(inputs, name, spec):
    """Parse a HasTraits object into a Numpy-style docstring."""
    desc_lines = []
    if spec.desc:
        desc = ''.join([spec.desc[0].capitalize(), spec.desc[1:]])
        if not desc.endswith('.') and (not desc.endswith('\n')):
            desc = '%s.' % desc
        desc_lines += desc.splitlines()
    argstr = spec.argstr
    if argstr and argstr.strip():
        pos = spec.position
        if pos is None:
            desc_lines += ['Maps to a command-line argument: :code:`{arg}`.'.format(arg=argstr.strip())]
        else:
            desc_lines += ['Maps to a command-line argument: :code:`{arg}` (position: {pos}).'.format(arg=argstr.strip(), pos=pos)]
    xor = spec.xor
    if xor:
        desc_lines += ['Mutually **exclusive** with inputs: %s.' % ', '.join(['``%s``' % x for x in xor])]
    requires = spec.requires
    if requires:
        desc_lines += ['**Requires** inputs: %s.' % ', '.join(['``%s``' % x for x in requires])]
    if spec.usedefault:
        default = spec.default_value()[1]
        if isinstance(default, (bytes, str)) and (not default):
            default = '""'
        desc_lines += ['(Nipype **default** value: ``%s``)' % str(default)]
    out_rst = ['{name} : {type}'.format(name=name, type=spec.full_info(inputs, name, None))]
    out_rst += _indent(desc_lines, 4)
    return out_rst