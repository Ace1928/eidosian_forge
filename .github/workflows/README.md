# 🚀 Eidosian Forge CI/CD Workflows

## Overview

This directory contains comprehensive GitHub Actions workflows for the Eidosian Forge repository. The workflows are designed to be robust, flexible, reusable, and provide detailed feedback for debugging and improvement.

## 📋 Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EIDOSIAN CI/CD PIPELINE                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────┐                          ┌──────────────┐
│  FORMAT (Auto)│                          │     LINT     │
│  • Black      │                          │  • Ruff      │
│  • isort      │                          │  • Flake8    │
│  • Prettier   │                          │  • ESLint    │
│  • Ruff Fix   │                          │  • MyPy      │
└───────┬───────┘                          └──────┬───────┘
        │                                         │
        └──────────────────┬──────────────────────┘
                           ▼
                    ┌─────────────┐
                    │    TEST     │
                    │  • Python   │
                    │  • TypeScript│
                    │  • Coverage │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │    BUILD    │
                    │  • Package  │
                    │  • Validate │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ INTEGRATION │
                    │  • Deploy   │
                    │  • Verify   │
                    └─────────────┘
```

## 📁 Workflow Files

### Primary Workflows

#### 1. **ci.yml** - Main CI Pipeline (v4.0.1)
*Comprehensive continuous integration pipeline*

**Triggers:**
- Push to `main`, `master`, `develop` branches
- Pull requests to protected branches
- Manual workflow dispatch

**Jobs:**
1. **Validate** - Determines what needs to run (Python/TypeScript detection)
2. **Format Check** - Ensures code is properly formatted
3. **Lint** - Comprehensive code quality checks
4. **Test** - Runs test suites for Python (3.10, 3.11, 3.12) and TypeScript
5. **Docs** - Builds documentation
6. **Build** - Creates distributable packages
7. **Integration** - Tests installed packages
8. **Pipeline** - Reports overall status

**Key Features:**
- Multi-version Python testing (3.10, 3.11, 3.12)
- Cross-platform testing (Ubuntu + Windows)
- Automatic language detection (runs only what changed)
- Detailed job summaries with tables and metrics
- Codecov integration for coverage tracking

---

#### 2. **workflow-lint.yml** - Workflow Syntax Validation
*Validates all GitHub Actions workflows with `actionlint`*

**Triggers:**
- Push and pull requests that touch workflow/config files
- Manual workflow dispatch

**Jobs:**
1. **actionlint** - Runs `actionlint` against `.github/workflows/*.yml`

**Key Features:**
- Catches workflow expression/syntax issues before merge
- Prevents non-workflow YAML from breaking workflow linting

---

#### 3. **format.yml** - Auto-Format Code (v1.0.0)
*Automatically formats code and commits changes*

**Triggers:**
- Workflow dispatch (manual trigger)
- Pull requests
- Push to protected branches (for code files only)

**Jobs:**
1. **format-python** - Formats Python with Black, isort, and Ruff
2. **format-typescript** - Formats TypeScript/JavaScript with Prettier
3. **format-summary** - Reports formatting results

**Key Features:**
- Commits formatting changes automatically (can be disabled)
- Skips CI on format commits (`[skip ci]`)
- Works on both Python and TypeScript/JavaScript
- Provides diff output for review

**Usage:**
```bash
# Trigger manually from GitHub Actions UI
# OR
# Let it run automatically on PRs/pushes
```

---

#### 4. **lint.yml** - Comprehensive Linting (v1.0.0)
*Detailed code quality checks with granular feedback*

**Triggers:**
- Push to protected branches (for code files)
- Pull requests
- Manual workflow dispatch

**Jobs:**
1. **lint-python** - Python linting with multiple tools
   - Black (formatting check)
   - isort (import sorting)
   - Ruff (fast linting)
   - Flake8 (style guide)
   - Pylint (deep analysis with score)
   - Mypy (type checking)

2. **lint-typescript** - TypeScript/JavaScript linting
   - Prettier (formatting)
   - ESLint (code quality)
   - TypeScript compiler (type checking)

3. **lint-summary** - Overall status with quick-fix commands

**Key Features:**
- Continue-on-error for each tool (all checks run)
- Detailed error counts and summaries
- Quick-fix commands in summary
- Pylint scoring
- GitHub annotations for inline errors

---

### Reusable Workflows

#### 5. **reusable-python-test.yml** - Reusable Python Testing
*Standardized Python testing workflow*

**Inputs:**
- `python-version` (default: "3.12")
- `test-path` (default: "tests/")
- `coverage-threshold` (default: 0)
- `install-command` (custom install)
- `pytest-args` (additional args)

**Outputs:**
- `coverage` - Coverage percentage
- `test-result` - Test outcome

**Example Usage:**
```yaml
jobs:
  test-module:
    uses: ./.github/workflows/reusable-python-test.yml
    with:
      python-version: "3.11"
      test-path: "my_module/tests"
      coverage-threshold: 80
      pytest-args: "-v --maxfail=1"
```

---

#### 6. **reusable-node-test.yml** - Reusable Node.js Testing
*Standardized TypeScript/JavaScript testing workflow*

**Inputs:**
- `node-version` (default: "20")
- `working-directory` (required)
- `test-command` (default: "npm test")
- `coverage-threshold` (default: 0)
- `run-lint` (default: true)
- `run-typecheck` (default: true)

**Outputs:**
- `coverage` - Coverage percentage
- `test-result` - Test outcome

**Example Usage:**
```yaml
jobs:
  test-frontend:
    uses: ./.github/workflows/reusable-node-test.yml
    with:
      node-version: "20"
      working-directory: "./game_forge/src/autoseed"
      coverage-threshold: 70
```

---

#### 7. **security-audit.yml** - Dependency Security Inventory
*Scheduled/manual Dependabot alert inventory with optional hard-fail gates*

**Triggers:**
- Daily scheduled run
- Manual workflow dispatch

**Inputs (manual dispatch):**
- `state` (`open|fixed|dismissed|auto_dismissed`)
- `fail_on_critical` (boolean)
- `fail_on_high` (boolean)

**Jobs:**
1. **dependabot-audit** - Collects alert summary via `scripts/dependabot_alert_inventory.py`
2. **remediation batch plan** - Builds deterministic vulnerability batches via `scripts/dependabot_remediation_plan.py`
3. **remediation issue sync** - Creates/updates/closes remediation batch issues via `scripts/sync_security_remediation_issues.py`
4. **publish summary** - Adds alert and remediation-batch counts to workflow summary
5. **optional gate enforcement** - Fails run when configured thresholds are exceeded
6. **artifact upload** - Stores JSON/Markdown/log reports in `reports/security/`

**Key Features:**
- Uses `security-events: read` permission for API access
- Produces report artifacts even if API access fails (fallback JSON)
- Supports strict/manual enforcement without forcing scheduled-run failures

---

#### 8. **workflow-action-pin-audit.yml** - Workflow Action Pin Drift Audit
*Detects workflow action reference drift against a committed lock file and enforces policy in CI*

**Triggers:**
- Push/PR when workflow files or pin-audit script/lock change
- Daily scheduled run
- Manual workflow dispatch

**Jobs:**
1. **audit** - Runs `scripts/audit_workflow_action_pins.py` with:
   - lock enforcement (`--enforce-lock`)
   - mutable-ref guard (`--fail-on-mutable`)
2. **artifact upload** - Publishes JSON audit output and the lock file

**Policy Model:**
- Lock file: `audit_data/workflow_action_lock.json`
- Enforced guarantees:
  - no unlocked workflow action refs
  - no lock drift for known action refs
  - no mutable branch refs (e.g. `@main`, `@master`)

---

## 🔧 Local Development Setup

### Workflow Config

Repository-wide workflow tuning values are stored in:

- `.github/workflow-config.yml`

This file is intentionally outside `.github/workflows/` so workflow linters only parse executable workflow files.

### Pre-commit Hooks

Install pre-commit hooks to automatically format and lint code before commits:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

The `.pre-commit-config.yaml` includes:
- Black (Python formatting)
- isort (import sorting)
- Ruff (linting + auto-fix)
- Prettier (TypeScript/JavaScript formatting)
- ESLint (JavaScript linting)
- Security checks (Bandit)
- File validation (trailing whitespace, large files, etc.)

---

## 🎯 Quick Commands

### Python

```bash
# Format code
black . --line-length=120
isort . --profile=black --line-length=120
ruff check . --fix

# Lint code
ruff check .
flake8 . --max-line-length=120
pylint --recursive=y .
mypy .

# Run tests
pytest --cov=. --cov-report=term-missing -v
```

### TypeScript/JavaScript

```bash
# Format code
prettier . --write

# Lint code
npm run lint  # or cd to specific project

# Type check
npm run typecheck

# Run tests
npm test
npm run test:coverage
```

---

## 📊 Status Badges

Add these to your README to show workflow status:

```markdown
[![CI](https://github.com/Ace1928/eidosian_forge/workflows/Eidosian%20Universal%20CI/badge.svg)](https://github.com/Ace1928/eidosian_forge/actions/workflows/ci.yml)
[![Lint](https://github.com/Ace1928/eidosian_forge/workflows/Lint%20Code/badge.svg)](https://github.com/Ace1928/eidosian_forge/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/Ace1928/eidosian_forge/branch/main/graph/badge.svg)](https://codecov.io/gh/Ace1928/eidosian_forge)
```

---

## 🔄 Workflow Customization

### Environment Variables

Common environment variables used across workflows:

```yaml
env:
  PYTHONUNBUFFERED: 1  # Real-time Python output
  FORCE_COLOR: 1       # Colored terminal output
  POETRY_VIRTUALENVS_CREATE: false  # Use system Python
  POETRY_VERSION: "1.8.3"  # Poetry version
```

### Workflow Dispatch Parameters

All workflows support manual triggering with optional parameters. Access via:
**Actions → [Workflow Name] → Run workflow**

---

## 🐛 Debugging Failed Workflows

### 1. Check Job Summary
- Click on the failed workflow run
- Review the "Summary" tab for high-level errors
- Check job-specific summaries for details
- Look for specific error messages in the annotations

### 2. View Detailed Logs
- Click on the failed job
- Expand log groups to see detailed output
- Look for `::error::` and `::warning::` annotations
- Each step shows its own exit code and output

### 3. Common Issues

#### "Maximum call stack size exceeded" during Python setup
- **Cause**: Large monorepo with many dependency files causes caching to fail
- **Fixed**: Removed pip caching to avoid this issue
- **Note**: Workflows may take slightly longer but are more reliable

#### Python setup fails
- Check the Python version is available (3.10, 3.11, 3.12)
- Verify `pyproject.toml` or `setup.py` is properly formatted
- Look for dependency conflicts in the logs

#### Tests fail
- Check the test output for specific failures
- Look for import errors or missing dependencies
- Verify test files are properly structured

### 4. Local Reproduction
```bash
# Run the same commands locally
# Example for lint failures:
black . --check --diff --line-length=120
ruff check .

# Example for test failures:
pytest -v
```

### 5. Use Workflow Dispatch
- Trigger workflows manually with debug flags
- Test specific scenarios without committing
- View logs in real-time as jobs run

---

## 📈 Coverage Reports

### Python Coverage
- Uploaded to Codecov automatically
- HTML reports available as artifacts
- View in workflow run: **Artifacts → coverage-html-py{version}**

### TypeScript Coverage
- Generated by Vitest with v8 provider
- Available as artifacts: **Artifacts → typescript-coverage**
- Summary displayed in job output

---

## 🔐 Security

### Secret Scanning
- Pre-commit hook detects private keys
- Bandit security scanner for Python
- Automated dependency updates (Dependabot - configure separately)

### Required Secrets
- `CODECOV_TOKEN` - For coverage upload (optional, set in repo secrets)
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

---

## 🚀 Performance Optimizations

1. **Caching**
   - pip dependencies cached by action
   - npm dependencies cached by action
   - Poetry virtualenvs cached

2. **Parallelization**
   - Multiple Python versions test in parallel
   - Independent jobs run concurrently
   - Matrix strategy for OS testing

3. **Concurrency Control**
   - Cancels duplicate runs on same branch
   - Prevents workflow queue buildup

4. **Conditional Execution**
   - Detects changed file types
   - Skips unnecessary jobs
   - `paths-ignore` for documentation

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Ruff Linter](https://docs.astral.sh/ruff/)
- [Prettier](https://prettier.io/)
- [Pytest](https://docs.pytest.org/)
- [Vitest](https://vitest.dev/)

---

## 🤝 Contributing

When modifying workflows:

1. Test changes in a fork first
2. Use workflow dispatch for testing
3. Document new inputs/outputs
4. Update this README
5. Version workflows (increment version in header comment)

---

## 📝 Changelog

### v4.0.1 (2026-01-24)
- 🐛 Fixed "Maximum call stack size exceeded" error by removing pip caching
- 📚 Added detailed troubleshooting section for common CI failures
- ⚡ Workflows now more reliable for large monorepos

### v4.0.0 (2026-01-24)
- ✨ Added comprehensive formatting workflow
- ✨ Added detailed linting workflow
- ✨ Added reusable testing workflows
- ✨ Added TypeScript/JavaScript support
- ✨ Enhanced CI pipeline with better summaries
- ✨ Added pre-commit configuration
- 📚 Created comprehensive documentation

### v3.15.0 (Previous)
- 🎯 Basic Python CI/CD
- 🧪 Multi-version Python testing
- 📦 Package building and validation

---

**Maintained by:** Eidosian Forge Team  
**Last Updated:** 2026-01-24
