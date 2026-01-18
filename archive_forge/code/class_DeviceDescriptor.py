class DeviceDescriptor(object):
    """Descriptor for basic attributes of the device."""
    usage_page = None
    usage = None
    vendor_id = None
    product_id = None
    product_string = None
    path = None
    internal_max_in_report_len = 0
    internal_max_out_report_len = 0

    def ToPublicDict(self):
        out = {}
        for k, v in list(self.__dict__.items()):
            if not k.startswith('internal_'):
                out[k] = v
        return out