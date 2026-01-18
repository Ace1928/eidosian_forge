import errno
import gyp.generator.ninja
import os
import re
import xml.sax.saxutils
def _TargetFromSpec(old_spec, params):
    """ Create fake target for xcode-ninja wrapper. """
    ninja_toplevel = None
    jobs = 0
    if params:
        options = params['options']
        ninja_toplevel = os.path.join(options.toplevel_dir, gyp.generator.ninja.ComputeOutputDir(params))
        jobs = params.get('generator_flags', {}).get('xcode_ninja_jobs', 0)
    target_name = old_spec.get('target_name')
    product_name = old_spec.get('product_name', target_name)
    product_extension = old_spec.get('product_extension')
    ninja_target = {}
    ninja_target['target_name'] = target_name
    ninja_target['product_name'] = product_name
    if product_extension:
        ninja_target['product_extension'] = product_extension
    ninja_target['toolset'] = old_spec.get('toolset')
    ninja_target['default_configuration'] = old_spec.get('default_configuration')
    ninja_target['configurations'] = {}
    new_xcode_settings = {}
    if ninja_toplevel:
        new_xcode_settings['CONFIGURATION_BUILD_DIR'] = '%s/$(CONFIGURATION)$(EFFECTIVE_PLATFORM_NAME)' % ninja_toplevel
    if 'configurations' in old_spec:
        for config in old_spec['configurations']:
            old_xcode_settings = old_spec['configurations'][config].get('xcode_settings', {})
            if 'IPHONEOS_DEPLOYMENT_TARGET' in old_xcode_settings:
                new_xcode_settings['CODE_SIGNING_REQUIRED'] = 'NO'
                new_xcode_settings['IPHONEOS_DEPLOYMENT_TARGET'] = old_xcode_settings['IPHONEOS_DEPLOYMENT_TARGET']
            for key in ['BUNDLE_LOADER', 'TEST_HOST']:
                if key in old_xcode_settings:
                    new_xcode_settings[key] = old_xcode_settings[key]
            ninja_target['configurations'][config] = {}
            ninja_target['configurations'][config]['xcode_settings'] = new_xcode_settings
    ninja_target['mac_bundle'] = old_spec.get('mac_bundle', 0)
    ninja_target['mac_xctest_bundle'] = old_spec.get('mac_xctest_bundle', 0)
    ninja_target['ios_app_extension'] = old_spec.get('ios_app_extension', 0)
    ninja_target['ios_watchkit_extension'] = old_spec.get('ios_watchkit_extension', 0)
    ninja_target['ios_watchkit_app'] = old_spec.get('ios_watchkit_app', 0)
    ninja_target['type'] = old_spec['type']
    if ninja_toplevel:
        ninja_target['actions'] = [{'action_name': 'Compile and copy %s via ninja' % target_name, 'inputs': [], 'outputs': [], 'action': ['env', 'PATH=%s' % os.environ['PATH'], 'ninja', '-C', new_xcode_settings['CONFIGURATION_BUILD_DIR'], target_name], 'message': 'Compile and copy %s via ninja' % target_name}]
        if jobs > 0:
            ninja_target['actions'][0]['action'].extend(('-j', jobs))
    return ninja_target