import operator
Return pair of resource attributes and corresponding display names.

    :param item: a dictionary which represents a resource.
        Keys of the dictionary are expected to be attributes of the resource.
        Values are not referred to by this method.

        .. code-block:: python

           {'id': 'myid', 'name': 'myname',
            'foo': 'bar', 'tenant_id': 'mytenan'}

    :param attr_map: a list of mapping from attribute to display name.
        The same format is used as for get_column_definitions attr_map.

        .. code-block:: python

           (('id', 'ID', LIST_BOTH),
            ('name', 'Name', LIST_BOTH),
            ('tenant_id', 'Project', LIST_LONG_ONLY))

    :return: A pair of tuple of attributes and tuple of display names.

        .. code-block:: python

           (('id', 'name', 'tenant_id', 'foo'),  # attributes
            ('ID', 'Name', 'Project', 'foo')     # display names

        Both tuples of attributes and display names are sorted by display names
        in the alphabetical order.
        Attributes not found in a given attr_map are kept as-is.
    