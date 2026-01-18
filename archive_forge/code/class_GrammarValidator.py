from __future__ import unicode_literals
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.document import Document
from .compiler import _CompiledGrammar
class GrammarValidator(Validator):
    """
    Validator which can be used for validation according to variables in
    the grammar. Each variable can have its own validator.

    :param compiled_grammar: `GrammarCompleter` instance.
    :param validators: `dict` mapping variable names of the grammar to the
                       `Validator` instances to be used for each variable.
    """

    def __init__(self, compiled_grammar, validators):
        assert isinstance(compiled_grammar, _CompiledGrammar)
        assert isinstance(validators, dict)
        self.compiled_grammar = compiled_grammar
        self.validators = validators

    def validate(self, document):
        m = self.compiled_grammar.match(document.text)
        if m:
            for v in m.variables():
                validator = self.validators.get(v.varname)
                if validator:
                    unwrapped_text = self.compiled_grammar.unescape(v.varname, v.value)
                    inner_document = Document(unwrapped_text, len(unwrapped_text))
                    try:
                        validator.validate(inner_document)
                    except ValidationError as e:
                        raise ValidationError(cursor_position=v.start + e.cursor_position, message=e.message)
        else:
            raise ValidationError(cursor_position=len(document.text), message='Invalid command')