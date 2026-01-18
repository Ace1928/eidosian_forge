import gyp.easy_xml as easy_xml
def AddCustomBuildRule(self, name, cmd, description, additional_dependencies, outputs, extensions):
    """Adds a rule to the tool file.

    Args:
      name: Name of the rule.
      description: Description of the rule.
      cmd: Command line of the rule.
      additional_dependencies: other files which may trigger the rule.
      outputs: outputs of the rule.
      extensions: extensions handled by the rule.
    """
    rule = ['CustomBuildRule', {'Name': name, 'ExecutionDescription': description, 'CommandLine': cmd, 'Outputs': ';'.join(outputs), 'FileExtensions': ';'.join(extensions), 'AdditionalDependencies': ';'.join(additional_dependencies)}]
    self.rules_section.append(rule)