from collections import ChainMap
from itemloaders.common import wrap_loader_context
from itemloaders.utils import arg_to_iter
class SelectJmes:
    """
    Query the input string for the jmespath (given at instantiation), and return the answer
    Requires : jmespath(https://github.com/jmespath/jmespath)
    Note: SelectJmes accepts only one input element at a time.

    Example:

    >>> from itemloaders.processors import SelectJmes, Compose, MapCompose
    >>> proc = SelectJmes("foo") #for direct use on lists and dictionaries
    >>> proc({'foo': 'bar'})
    'bar'
    >>> proc({'foo': {'bar': 'baz'}})
    {'bar': 'baz'}

    Working with Json:

    >>> import json
    >>> proc_single_json_str = Compose(json.loads, SelectJmes("foo"))
    >>> proc_single_json_str('{"foo": "bar"}')
    'bar'
    >>> proc_json_list = Compose(json.loads, MapCompose(SelectJmes('foo')))
    >>> proc_json_list('[{"foo":"bar"}, {"baz":"tar"}]')
    ['bar']
    """

    def __init__(self, json_path):
        self.json_path = json_path
        import jmespath
        self.compiled_path = jmespath.compile(self.json_path)

    def __call__(self, value):
        """Query value for the jmespath query and return answer
        :param value: a data structure (dict, list) to extract from
        :return: Element extracted according to jmespath query
        """
        return self.compiled_path.search(value)