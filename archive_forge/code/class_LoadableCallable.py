import traitlets
class LoadableCallable(traitlets.TraitType):
    """A trait which (maybe) loads a callable."""
    info_text = 'a loadable callable'

    def validate(self, obj, value):
        if isinstance(value, str):
            try:
                value = traitlets.import_item(value)
            except Exception:
                self.error(obj, value)
        if callable(value):
            return value
        else:
            self.error(obj, value)