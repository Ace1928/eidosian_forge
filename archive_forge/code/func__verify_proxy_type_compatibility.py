def _verify_proxy_type_compatibility(self, compatible_proxy):
    if self.proxyType not in (ProxyType.UNSPECIFIED, compatible_proxy):
        raise ValueError(f'Specified proxy type ({compatible_proxy}) not compatible with current setting ({self.proxyType})')