import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXNativeTarget(XCTarget):
    _schema = XCTarget._schema.copy()
    _schema.update({'buildPhases': [1, XCBuildPhase, 1, 1, [PBXSourcesBuildPhase(), PBXFrameworksBuildPhase()]], 'buildRules': [1, PBXBuildRule, 1, 1, []], 'productReference': [0, PBXFileReference, 0, 1], 'productType': [0, str, 0, 1]})
    _product_filetypes = {'com.apple.product-type.application': ['wrapper.application', '', '.app'], 'com.apple.product-type.application.watchapp': ['wrapper.application', '', '.app'], 'com.apple.product-type.watchkit-extension': ['wrapper.app-extension', '', '.appex'], 'com.apple.product-type.app-extension': ['wrapper.app-extension', '', '.appex'], 'com.apple.product-type.bundle': ['wrapper.cfbundle', '', '.bundle'], 'com.apple.product-type.framework': ['wrapper.framework', '', '.framework'], 'com.apple.product-type.library.dynamic': ['compiled.mach-o.dylib', 'lib', '.dylib'], 'com.apple.product-type.library.static': ['archive.ar', 'lib', '.a'], 'com.apple.product-type.tool': ['compiled.mach-o.executable', '', ''], 'com.apple.product-type.bundle.unit-test': ['wrapper.cfbundle', '', '.xctest'], 'com.apple.product-type.bundle.ui-testing': ['wrapper.cfbundle', '', '.xctest'], 'com.googlecode.gyp.xcode.bundle': ['compiled.mach-o.dylib', '', '.so'], 'com.apple.product-type.kernel-extension': ['wrapper.kext', '', '.kext']}

    def __init__(self, properties=None, id=None, parent=None, force_outdir=None, force_prefix=None, force_extension=None):
        XCTarget.__init__(self, properties, id, parent)
        if 'productName' in self._properties and 'productType' in self._properties and ('productReference' not in self._properties) and (self._properties['productType'] in self._product_filetypes):
            products_group = None
            pbxproject = self.PBXProjectAncestor()
            if pbxproject is not None:
                products_group = pbxproject.ProductsGroup()
            if products_group is not None:
                filetype, prefix, suffix = self._product_filetypes[self._properties['productType']]
                if self._properties['productType'] == 'com.googlecode.gyp.xcode.bundle':
                    self._properties['productType'] = 'com.apple.product-type.library.dynamic'
                    self.SetBuildSetting('MACH_O_TYPE', 'mh_bundle')
                    self.SetBuildSetting('DYLIB_CURRENT_VERSION', '')
                    self.SetBuildSetting('DYLIB_COMPATIBILITY_VERSION', '')
                    if force_extension is None:
                        force_extension = suffix[1:]
                if self._properties['productType'] == 'com.apple.product-type-bundle.unit.test' or self._properties['productType'] == 'com.apple.product-type-bundle.ui-testing':
                    if force_extension is None:
                        force_extension = suffix[1:]
                if force_extension is not None:
                    suffix = '.' + force_extension
                    if filetype.startswith('wrapper.'):
                        self.SetBuildSetting('WRAPPER_EXTENSION', force_extension)
                    else:
                        self.SetBuildSetting('EXECUTABLE_EXTENSION', force_extension)
                    if filetype.startswith('compiled.mach-o.executable'):
                        product_name = self._properties['productName']
                        product_name += suffix
                        suffix = ''
                        self.SetProperty('productName', product_name)
                        self.SetBuildSetting('PRODUCT_NAME', product_name)
                if force_prefix is not None:
                    prefix = force_prefix
                if filetype.startswith('wrapper.'):
                    self.SetBuildSetting('WRAPPER_PREFIX', prefix)
                else:
                    self.SetBuildSetting('EXECUTABLE_PREFIX', prefix)
                if force_outdir is not None:
                    self.SetBuildSetting('TARGET_BUILD_DIR', force_outdir)
                product_name = self._properties['productName']
                prefix_len = len(prefix)
                if prefix_len and product_name[:prefix_len] == prefix:
                    product_name = product_name[prefix_len:]
                    self.SetProperty('productName', product_name)
                    self.SetBuildSetting('PRODUCT_NAME', product_name)
                ref_props = {'explicitFileType': filetype, 'includeInIndex': 0, 'path': prefix + product_name + suffix, 'sourceTree': 'BUILT_PRODUCTS_DIR'}
                file_ref = PBXFileReference(ref_props)
                products_group.AppendChild(file_ref)
                self.SetProperty('productReference', file_ref)

    def GetBuildPhaseByType(self, type):
        if 'buildPhases' not in self._properties:
            return None
        the_phase = None
        for phase in self._properties['buildPhases']:
            if isinstance(phase, type):
                assert the_phase is None
                the_phase = phase
        return the_phase

    def HeadersPhase(self):
        headers_phase = self.GetBuildPhaseByType(PBXHeadersBuildPhase)
        if headers_phase is None:
            headers_phase = PBXHeadersBuildPhase()
            insert_at = len(self._properties['buildPhases'])
            for index, phase in enumerate(self._properties['buildPhases']):
                if isinstance(phase, PBXResourcesBuildPhase) or isinstance(phase, PBXSourcesBuildPhase) or isinstance(phase, PBXFrameworksBuildPhase):
                    insert_at = index
                    break
            self._properties['buildPhases'].insert(insert_at, headers_phase)
            headers_phase.parent = self
        return headers_phase

    def ResourcesPhase(self):
        resources_phase = self.GetBuildPhaseByType(PBXResourcesBuildPhase)
        if resources_phase is None:
            resources_phase = PBXResourcesBuildPhase()
            insert_at = len(self._properties['buildPhases'])
            for index, phase in enumerate(self._properties['buildPhases']):
                if isinstance(phase, PBXSourcesBuildPhase) or isinstance(phase, PBXFrameworksBuildPhase):
                    insert_at = index
                    break
            self._properties['buildPhases'].insert(insert_at, resources_phase)
            resources_phase.parent = self
        return resources_phase

    def SourcesPhase(self):
        sources_phase = self.GetBuildPhaseByType(PBXSourcesBuildPhase)
        if sources_phase is None:
            sources_phase = PBXSourcesBuildPhase()
            self.AppendProperty('buildPhases', sources_phase)
        return sources_phase

    def FrameworksPhase(self):
        frameworks_phase = self.GetBuildPhaseByType(PBXFrameworksBuildPhase)
        if frameworks_phase is None:
            frameworks_phase = PBXFrameworksBuildPhase()
            self.AppendProperty('buildPhases', frameworks_phase)
        return frameworks_phase

    def AddDependency(self, other):
        XCTarget.AddDependency(self, other)
        static_library_type = 'com.apple.product-type.library.static'
        shared_library_type = 'com.apple.product-type.library.dynamic'
        framework_type = 'com.apple.product-type.framework'
        if isinstance(other, PBXNativeTarget) and 'productType' in self._properties and (self._properties['productType'] != static_library_type) and ('productType' in other._properties) and (other._properties['productType'] == static_library_type or ((other._properties['productType'] == shared_library_type or other._properties['productType'] == framework_type) and (not other.HasBuildSetting('MACH_O_TYPE') or other.GetBuildSetting('MACH_O_TYPE') != 'mh_bundle'))):
            file_ref = other.GetProperty('productReference')
            pbxproject = self.PBXProjectAncestor()
            other_pbxproject = other.PBXProjectAncestor()
            if pbxproject != other_pbxproject:
                other_project_product_group = pbxproject.AddOrGetProjectReference(other_pbxproject)[0]
                file_ref = other_project_product_group.GetChildByRemoteObject(file_ref)
            self.FrameworksPhase().AppendProperty('files', PBXBuildFile({'fileRef': file_ref}))