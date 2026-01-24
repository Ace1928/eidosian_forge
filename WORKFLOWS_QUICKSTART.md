# 🎯 Quick Start Guide: CI/CD Workflows

## For Developers

### 🚀 Getting Started

1. **Install pre-commit hooks** (one-time setup):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. **Before every commit**, hooks will automatically:
   - Format your Python code with Black, isort, and Ruff
   - Format your TypeScript/JavaScript with Prettier
   - Check for security issues
   - Validate file syntax

3. **Before pushing**, run all checks locally:
   ```bash
   ./scripts/run-checks.sh
   ```

4. **To auto-fix issues**:
   ```bash
   ./scripts/run-checks.sh --fix
   ```

---

## Workflow Overview

### 1️⃣ When You Push or Create a PR

**Automatically runs:**
- ✅ Format checking
- 🔍 Linting (multiple tools)
- 🧪 Tests (Python 3.10, 3.11, 3.12 + TypeScript)
- 📚 Documentation build
- 📦 Package build
- 🧩 Integration tests

**You'll see:**
- Detailed summaries in the GitHub Actions tab
- Inline annotations on your code for issues
- Quick-fix suggestions in job summaries

### 2️⃣ Manual Workflows

**Format Your Code:**
1. Go to Actions → "Auto Format Code"
2. Click "Run workflow"
3. Changes are automatically committed with `[skip ci]`

**Run Linting Only:**
1. Go to Actions → "Lint Code"
2. Click "Run workflow"
3. View detailed reports in job summaries

---

## Common Scenarios

### ❌ My PR failed format checks

**Quick fix:**
```bash
# Run locally to see issues
./scripts/run-checks.sh

# Auto-fix formatting
./scripts/run-checks.sh --fix

# Commit and push
git add .
git commit -m "Fix formatting"
git push
```

**OR** trigger the auto-format workflow from GitHub Actions.

---

### ❌ My PR failed linting

**Check the job summary** for specific errors and suggestions.

**Common fixes:**
```bash
# Python
black . --line-length=120
isort . --profile=black
ruff check . --fix

# TypeScript/JavaScript
prettier . --write
cd game_forge/src/autoseed && npm run lint --fix
```

---

### ❌ Workflow fails with "Maximum call stack size exceeded"

**This issue has been fixed!** 

The error was caused by pip caching trying to scan too many dependency files in this large monorepo. We've removed pip caching from all workflows.

**If you still see this error:**
1. Re-run the workflow (it should work now)
2. Check that you're using the latest version of the workflows

---

### ❌ Tests are failing

1. **Run tests locally:**
   ```bash
   # Python
   pytest -v
   
   # TypeScript
   cd game_forge/src/autoseed
   npm test
   ```

2. **Check the test logs** in GitHub Actions for detailed error messages

3. **View coverage reports** in the Artifacts section

---

### 📝 I want to skip CI for a commit

Add `[skip ci]` to your commit message:
```bash
git commit -m "Update README [skip ci]"
```

⚠️ **Note:** Format and lint workflows still run on PRs regardless of `[skip ci]`

---

## Workflow Files Reference

| Workflow | Purpose | Trigger | Can Skip? |
|----------|---------|---------|-----------|
| `ci.yml` | Main CI/CD pipeline | Push/PR | Yes* |
| `format.yml` | Auto-format code | Push/PR/Manual | No on PR |
| `lint.yml` | Detailed linting | Push/PR/Manual | No on PR |
| `reusable-python-test.yml` | Reusable Python testing | Called by other workflows | N/A |
| `reusable-node-test.yml` | Reusable Node testing | Called by other workflows | N/A |

*\*Respects `[skip ci]` on push, always runs on PR*

---

## Configuration Files

| File | Purpose |
|------|---------|
| `.pre-commit-config.yaml` | Local git hooks configuration |
| `.github/workflows/config.yml` | Reference for workflow settings |
| `ruff.toml` | Ruff linter configuration |
| `pytest.ini` | Pytest configuration |
| `game_forge/src/autoseed/eslint.config.js` | ESLint configuration |

---

## Tips & Tricks

### 💡 Speed up local checks

Run only what you changed:
```bash
# Python only
./scripts/run-checks.sh --python-only

# TypeScript only
./scripts/run-checks.sh --ts-only
```

### 💡 View detailed logs

```bash
# Pre-commit
pre-commit run --all-files --verbose

# Individual tools
black . --check --verbose
ruff check . --show-files
```

### 💡 Update pre-commit hooks

```bash
pre-commit autoupdate
```

### 💡 Bypass pre-commit hooks (not recommended)

```bash
git commit --no-verify -m "Emergency fix"
```

---

## Troubleshooting

### Problem: Pre-commit hooks are slow

**Solution:** They only run on changed files. For speed:
```bash
# Run on staged files only
git add <files>
pre-commit run

# Skip specific hooks
SKIP=pylint git commit -m "message"
```

### Problem: Hooks fail on files I didn't touch

**Solution:** This is normal if the repository had pre-existing issues. Fix them:
```bash
# Auto-fix everything
./scripts/run-checks.sh --fix

# Or commit the formatted files
git add .
git commit -m "Fix formatting across repository"
```

### Problem: Can't find the workflow run

1. Go to repository → Actions tab
2. Select the workflow name (e.g., "Lint Code")
3. Find your branch/commit in the list
4. Click to view details

---

## Getting Help

- 📖 **Full Documentation:** [.github/workflows/README.md](.github/workflows/README.md)
- 🐛 **Report Issues:** Open a GitHub issue
- 💬 **Ask Questions:** Use GitHub Discussions

---

**Last Updated:** 2026-01-24  
**Maintained by:** Eidosian Forge Team
