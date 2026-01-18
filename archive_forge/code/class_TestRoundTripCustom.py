import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
class TestRoundTripCustom:

    def test_X1(self):
        register_xxx()
        round_trip('        !xxx\n        name: Anthon\n        location: Germany\n        language: python\n        ')

    @pytest.mark.xfail(strict=True)
    def test_X_pre_tag_comment(self):
        register_xxx()
        round_trip('        -\n          # hello\n          !xxx\n          name: Anthon\n          location: Germany\n          language: python\n        ')

    @pytest.mark.xfail(strict=True)
    def test_X_post_tag_comment(self):
        register_xxx()
        round_trip('        - !xxx\n          # hello\n          name: Anthon\n          location: Germany\n          language: python\n        ')

    def test_scalar_00(self):
        round_trip('        Outputs:\n          Vpc:\n            Value: !Ref: vpc    # first tag\n            Export:\n              Name: !Sub "${AWS::StackName}-Vpc"  # second tag\n        ')