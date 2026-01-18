smoke:
	./scripts/bootstrap.sh
	.venv/bin/python -m core.config --dir cfg --validate
	bin/eidctl state --migrate
	echo "hello" | bin/eidctl journal --add - --type note --tags smoke
	.venv/bin/python -c "from core import events as E, db as DB; E.append('state','smoke.event',{'ok':True}); DB.insert_metric('state','smoke.metric',42.0); DB.insert_journal('state','smoke.db','ok'); print('ok')"
	bin/eidosd --state-dir state --once
	bin/eidctl state --json | jq -e '.totals.note >= 1 and .files.bus >= 1'
	.venv/bin/python -m pytest -q

.PHONY: loop
loop:
	bin/eidosd --state-dir state --loop --tick 1 --max-beats 5
	# quick health:
	sqlite3 state/e3.sqlite "select key,count(*) from metrics group by key order by key;"

.PHONY: loop-prune
loop-prune:
	bin/eidosd --state-dir state --loop --tick 0.5 --max-beats 300
	sqlite3 state/e3.sqlite "select key,count(*) from metrics group by key order by key;"
