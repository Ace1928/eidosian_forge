from pbr.hooks import base
from pbr import packaging
class BackwardsCompatConfig(base.BaseConfig):
    section = 'backwards_compat'

    def hook(self):
        self.config['include_package_data'] = 'True'
        packaging.append_text_list(self.config, 'dependency_links', packaging.parse_dependency_links())
        packaging.append_text_list(self.config, 'tests_require', packaging.parse_requirements(packaging.TEST_REQUIREMENTS_FILES, strip_markers=True))