from __future__ import annotations

from agent_forge.consciousness.protocol import (
    default_preregistration,
    default_rac_ap_protocol,
    validate_rac_ap_protocol,
)


def test_default_protocol_includes_rac_ap_sections() -> None:
    protocol = default_rac_ap_protocol()
    result = validate_rac_ap_protocol(protocol)
    assert result.valid is True
    assert isinstance(protocol.get("falsification_criteria"), list)
    assert isinstance(protocol.get("perturbation_suite"), dict)
    assert isinstance(protocol.get("external_benchmarks"), dict)
    assert isinstance(protocol.get("security_evaluation"), dict)
    assert isinstance(protocol.get("construct_validity"), dict)


def test_protocol_validation_rejects_missing_core_sections() -> None:
    protocol = default_rac_ap_protocol()
    protocol["perturbation_suite"] = {"required_recipes": []}
    protocol["external_benchmarks"] = {"required": [], "repeats_per_seed": 0}
    protocol["security_evaluation"] = {"max_attack_success_rate": 2.0}
    protocol["construct_validity"] = {"minimum_negative_controls": 0}
    protocol["falsification_criteria"] = []
    result = validate_rac_ap_protocol(protocol)
    assert result.valid is False
    joined = "\n".join(result.errors)
    assert "falsification criterion" in joined
    assert "perturbation_suite.required_recipes" in joined
    assert "external_benchmarks.required" in joined
    assert "max_attack_success_rate" in joined
    assert "minimum_negative_controls" in joined


def test_preregistration_includes_protocol_sections() -> None:
    protocol = default_rac_ap_protocol()
    prereg = default_preregistration(
        protocol=protocol,
        study_name="rac_ap_cycle",
        hypothesis="Winner-linked ignition improves intervention outcomes.",
        owner="eidos",
    )
    assert prereg.get("falsification_criteria")
    assert isinstance(prereg.get("perturbation_suite"), dict)
    assert isinstance(prereg.get("external_benchmarks"), dict)
    assert isinstance(prereg.get("security_evaluation"), dict)
    assert isinstance(prereg.get("construct_validity"), dict)
