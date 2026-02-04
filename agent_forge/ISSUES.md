# Issues: agent_forge

**Last Updated**: 2026-01-25

---

## 🔴 Critical

### ISSUE-001: CLI Scripts Missing
**Description**: Tests expect `bin/eidctl`, `bin/eidosd`, `bin/eidtop` but these don't exist. Scripts are now installed via entry_points.
**Impact**: 10+ test failures
**Status**: 🔶 Needs migration

### ISSUE-002: Task API Changed
**Description**: `Task.__init__()` now requires `task_id` positional arg but tests use old signature.
**Impact**: 4 test failures
**Status**: 🔶 Needs fix

---

## 🟠 High Priority

### ISSUE-003: Directory Structure Confusion
**Description**: Split between `core/` and `src/` unclear.
**Impact**: Developer confusion, import complexity.
**Status**: 🔶 Needs investigation

### ISSUE-004: Documentation Incomplete
**Description**: README and API docs need work.
**Status**: 🔶 Needs work

---

## 🟡 Medium Priority

### ISSUE-005: Ollama Dependency
**Description**: Some tests require Ollama running.
**Impact**: CI may fail without setup.
**Status**: ⬜ Planned

### ISSUE-006: Single Agent Only
**Description**: No multi-agent coordination yet.
**Status**: ⬜ Planned

---

## 🟢 Low Priority

### ISSUE-007: TUI Polish
**Description**: eidtop could use visual improvements.
**Status**: ⬜ Future

---

## 📊 Test Status

| Category | Count |
|----------|-------|
| Passed | 37 |
| Failed | 10 |
| **Coverage** | ~70% |

---

*Track issues. Fix issues. Forge agents.*
