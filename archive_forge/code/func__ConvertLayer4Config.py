def _ConvertLayer4Config(layer4_configs):
    """Converts Firewall Policy Layer4 configs to terraform script.

  Args:
    layer4_configs: Firewall Policy layer4 configs

  Returns:
    Terraform script
  """
    records = []
    template = '    layer4_configs {{\n      ip_protocol = "{ip_protocol}"\n      ports = [{ports}]\n    }}\n'
    for config in layer4_configs:
        records.append(template.format(ip_protocol=config.ipProtocol, ports=_ConvertArray(config.ports)))
    return ''.join(records)