import ast
from nbconvert.preprocessors import Preprocessor
class OptsMagicProcessor(Preprocessor):
    """
    Preprocessor to convert notebooks to Python source to convert use of
    opts magic to use the util.opts utility instead.
    """

    def preprocess_cell(self, cell, resources, index):
        if cell['cell_type'] == 'code':
            source = replace_line_magic(cell['source'], '%opts', template='hv.util.opts({line!r})')
            source, opts_lines = filter_magic(source, '%%opts')
            if opts_lines:
                template = 'hv.util.opts({options!r}, {{expr}})'.format(options=' '.join(opts_lines).replace('{', '{{').replace('}', '}}'))
                source = wrap_cell_expression(source, template)
            cell['source'] = source
        return (cell, resources)

    def __call__(self, nb, resources):
        return self.preprocess(nb, resources)