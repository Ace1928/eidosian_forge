from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class CompositionConfig(VegaLiteSchema):
    """CompositionConfig schema wrapper

    Parameters
    ----------

    columns : float
        The number of columns to include in the view composition layout.

        **Default value** : ``undefined`` -- An infinite number of columns (a single row)
        will be assumed. This is equivalent to ``hconcat`` (for ``concat`` ) and to using
        the ``column`` channel (for ``facet`` and ``repeat`` ).

        **Note** :

        1) This property is only for:


        * the general (wrappable) ``concat`` operator (not ``hconcat`` / ``vconcat`` )
        * the ``facet`` and ``repeat`` operator with one field/repetition definition
          (without row/column nesting)

        2) Setting the ``columns`` to ``1`` is equivalent to ``vconcat`` (for ``concat`` )
        and to using the ``row`` channel (for ``facet`` and ``repeat`` ).
    spacing : float
        The default spacing in pixels between composed sub-views.

        **Default value** : ``20``
    """
    _schema = {'$ref': '#/definitions/CompositionConfig'}

    def __init__(self, columns: Union[float, UndefinedType]=Undefined, spacing: Union[float, UndefinedType]=Undefined, **kwds):
        super(CompositionConfig, self).__init__(columns=columns, spacing=spacing, **kwds)