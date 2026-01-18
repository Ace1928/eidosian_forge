from collections import abc
class MultiView(object):
    """A Text View Containing Multiple Views

    This view simply serializes each
    value in the data model, and then
    joins them with newlines (ignoring
    the key values altogether).  This is
    useful for serializing lists of models
    (as array-like dicts).
    """

    def __call__(self, model):
        res = sorted([str(model[key]) for key in model])
        return '\n'.join(res)