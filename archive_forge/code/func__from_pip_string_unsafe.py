import itertools
import operator
import sys
@classmethod
def _from_pip_string_unsafe(klass, version_string):
    version_string = version_string.lstrip('vV')
    if not version_string[:1].isdigit():
        raise ValueError('Invalid version %r' % version_string)
    input_components = version_string.split('.')
    components = [c for c in input_components if c.isdigit()]
    digit_len = len(components)
    if digit_len == 0:
        raise ValueError('Invalid version %r' % version_string)
    elif digit_len < 3:
        if digit_len < len(input_components) and input_components[digit_len][0].isdigit():
            mixed_component = input_components[digit_len]
            last_component = ''.join(itertools.takewhile(lambda x: x.isdigit(), mixed_component))
            components.append(last_component)
            input_components[digit_len:digit_len + 1] = [last_component, mixed_component[len(last_component):]]
            digit_len += 1
        components.extend([0] * (3 - digit_len))
    components.extend(input_components[digit_len:])
    major = int(components[0])
    minor = int(components[1])
    dev_count = None
    post_count = None
    prerelease_type = None
    prerelease = None

    def _parse_type(segment):
        isdigit = operator.methodcaller('isdigit')
        segment = ''.join(itertools.dropwhile(isdigit, segment))
        isalpha = operator.methodcaller('isalpha')
        prerelease_type = ''.join(itertools.takewhile(isalpha, segment))
        prerelease = segment[len(prerelease_type):]
        return (prerelease_type, int(prerelease))
    if _is_int(components[2]):
        patch = int(components[2])
    else:
        patch = 0
        components[2:2] = [0]
    remainder = components[3:]
    remainder_starts_with_int = False
    try:
        if remainder and int(remainder[0]):
            remainder_starts_with_int = True
    except ValueError:
        pass
    if remainder_starts_with_int:
        dev_count = int(remainder[0])
    else:
        if remainder and (remainder[0][0] == '0' or remainder[0][0] in ('a', 'b', 'r')):
            prerelease_type, prerelease = _parse_type(remainder[0])
            remainder = remainder[1:]
        while remainder:
            component = remainder[0]
            if component.startswith('dev'):
                dev_count = int(component[3:])
            elif component.startswith('post'):
                dev_count = None
                post_count = int(component[4:])
            else:
                raise ValueError('Unknown remainder %r in %r' % (remainder, version_string))
            remainder = remainder[1:]
    result = SemanticVersion(major, minor, patch, prerelease_type=prerelease_type, prerelease=prerelease, dev_count=dev_count)
    if post_count:
        if dev_count:
            raise ValueError('Cannot combine postN and devN - no mapping in %r' % (version_string,))
        result = result.increment().to_dev(post_count)
    return result