import decimal
from .draft04 import CodeGeneratorDraft04, JSON_TYPE_TO_PYTHON_TYPE
from .exceptions import JsonSchemaDefinitionException
from .generator import enforce_list

        Means that value is valid when is equeal to const definition.

        .. code-block:: python

            {
                'const': 42,
            }

        Only valid value is 42 in this example.
        