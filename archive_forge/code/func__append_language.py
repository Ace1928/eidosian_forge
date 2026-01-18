from enum import Enum
def _append_language(self, language_field, language):
    """Append LANGUAGE_FIELD and LANGUAGE."""
    if language_field is not None:
        self.args.append('LANGUAGE_FIELD')
        self.args.append(language_field)
    if language is not None:
        self.args.append('LANGUAGE')
        self.args.append(language)