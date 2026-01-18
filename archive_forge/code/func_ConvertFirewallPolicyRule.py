def ConvertFirewallPolicyRule(rule):
    """Converts Firewall Policy rule to terraform script.

  Args:
    rule: Firewall Policy rule

  Returns:
    Terraform script
  """
    return 'resource "google_compute_network_firewall_policy_rule" "auto_generated_rule_{priority}" {{\n  action                  = "{action}"\n  description             = "{description}"\n  direction               = "{direction}"\n  disabled                = {disabled}\n  enable_logging          = {enable_logging}\n  firewall_policy         = google_compute_network_firewall_policy.auto_generated_firewall_policy.name\n  priority                = {priority}\n  rule_name               = "{rule_name}"\n\n  match {{\n    dest_ip_ranges = [{dest_ip_ranges}]\n    src_ip_ranges = [{src_ip_ranges}]\n{src_secure_tags}{layer4_configs}  }}\n{target_secure_tags}}}\n'.format(action=rule.action, description=rule.description, direction=rule.direction, disabled=_ConvertBoolean(rule.disabled), enable_logging=_ConvertBoolean(rule.enableLogging), priority=rule.priority, rule_name=rule.ruleName, dest_ip_ranges=_ConvertArray(rule.match.destIpRanges), src_ip_ranges=_ConvertArray(rule.match.srcIpRanges), src_secure_tags=_ConvertSrcTags(rule.match.srcSecureTags), target_secure_tags=_ConvertTargetTags(rule.targetSecureTags), layer4_configs=_ConvertLayer4Config(rule.match.layer4Configs))