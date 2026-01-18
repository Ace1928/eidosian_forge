class ProxyType:
    """Set of possible types of proxy.

    Each proxy type has 2 properties:    'ff_value' is value of Firefox
    profile preference,    'string' is id of proxy type.
    """
    DIRECT = ProxyTypeFactory.make(0, 'DIRECT')
    MANUAL = ProxyTypeFactory.make(1, 'MANUAL')
    PAC = ProxyTypeFactory.make(2, 'PAC')
    RESERVED_1 = ProxyTypeFactory.make(3, 'RESERVED1')
    AUTODETECT = ProxyTypeFactory.make(4, 'AUTODETECT')
    SYSTEM = ProxyTypeFactory.make(5, 'SYSTEM')
    UNSPECIFIED = ProxyTypeFactory.make(6, 'UNSPECIFIED')

    @classmethod
    def load(cls, value):
        if isinstance(value, dict) and 'string' in value:
            value = value['string']
        value = str(value).upper()
        for attr in dir(cls):
            attr_value = getattr(cls, attr)
            if isinstance(attr_value, dict) and 'string' in attr_value and (attr_value['string'] == value):
                return attr_value
        raise Exception(f'No proxy type is found for {value}')