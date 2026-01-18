def _prettify_message(message):
    """Kombu doesn't currently have a useful ``__str__()`` or ``__repr__()``.

    This provides something decent(ish) for debugging (or other purposes) so
    that messages are more nice and understandable....
    """
    if message.content_type is not None:
        properties = {'content_type': message.content_type}
    else:
        properties = {}
    for name in _MSG_PROPERTIES:
        segments = name.split('/')
        try:
            value = _get_deep(message.properties, *segments)
        except (KeyError, ValueError, TypeError):
            pass
        else:
            if value is not None:
                properties[segments[-1]] = value
    if message.body is not None:
        properties['body_length'] = len(message.body)
    return '%(delivery_tag)s: %(properties)s' % {'delivery_tag': message.delivery_tag, 'properties': properties}