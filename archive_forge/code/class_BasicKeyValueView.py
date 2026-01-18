from collections import abc
class BasicKeyValueView(object):
    """A Basic Key-Value Text View

    This view performs a naive serialization of a model into
    text using a basic key-value method, where each
    key-value pair is rendered as "key = str(value)"
    """

    def __call__(self, model):
        res = ''
        for key in sorted(model):
            res += '{key} = {value}\n'.format(key=key, value=model[key])
        return res